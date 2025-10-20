#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline principal de Constitutional AI.
"""

import os
import random
from typing import List, Dict, Any

from config import Args
from data_handler import (
    load_constitution, 
    load_red_teaming_prompts_from_parquet,
    save_constitution_with_results,
    save_parquet_dataset
)
from model_manager import (
    download_model_if_needed,
    load_tokenizer,
    initialize_vllm,
    create_sampling_params,
    run_vllm_generate
)
from utils import (
    set_seed,
    setup_vllm_environment,
    create_output_paths,
    build_chat_with_fewshots,
    chats_to_prompts,
    chunked,
    parse_gpu_config
)


class ConstitutionalAIPipeline:
    """Pipeline para Constitutional AI con procesamiento en lotes."""
    
    def __init__(self, args: Args):
        """
        Inicializa el pipeline.
        
        Args:
            args: configuración del pipeline
        """
        self.args = args
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.constitutions = []
        self.fewshots = []
        
    def setup(self):
        """Configura el pipeline: modelos, datos y entorno."""
        # Configurar semilla y entorno
        set_seed(self.args.seed)
        setup_vllm_environment(self.args.gpus)
        
        # Cargar constitución
        print(" Cargando constitución...")
        data = load_constitution(self.args.constitution_path)
        self.constitutions = data["constitutions"]
        self.fewshots = data.get("few_shot_examples", [])
        
        # Configurar modelo
        print(" Configurando modelo...")
        model_to_use = download_model_if_needed(self.args.model, self.args.model_path)
        self.tokenizer = load_tokenizer(model_to_use)
        self.llm = initialize_vllm(self.args, model_to_use)
        self.sampling_params = create_sampling_params(self.args, self.tokenizer)
        
        return data
    
    def load_red_teaming_prompts(self) -> List[str]:
        """Carga los prompts de red teaming."""
        if self.args.red_teaming_parquet and os.path.exists(self.args.red_teaming_parquet):
            print(f" Cargando red teaming prompts desde parquet: {self.args.red_teaming_parquet}")
            red_prompts = load_red_teaming_prompts_from_parquet(self.args.red_teaming_parquet)
        else:
            print(" Usando red teaming prompts por defecto")
            red_prompts = list(self.args.red_prompts)
        
        if self.args.max_samples != -1:
            red_prompts = red_prompts[:self.args.max_samples]
            print(f" Limitando a {len(red_prompts)} prompts (max_samples={self.args.max_samples})")
        else:
            print(f" Procesando todos los {len(red_prompts)} prompts disponibles")
            
        return red_prompts
    
    def stage_1_initial_response(self, red_prompts: List[str]) -> List[str]:
        """
        Etapa 1: Genera respuestas iniciales a los prompts de red teaming.
        
        Args:
            red_prompts: lista de prompts de red teaming
            
        Returns:
            Lista de respuestas iniciales
        """
        print(" Etapa 1: Generando respuestas iniciales...")
        
        # Construir chats con few-shots para mejor inferencia
        chats_stage1 = [build_chat_with_fewshots(self.fewshots, p) for p in red_prompts]
        
        # Procesar en lotes
        init_responses: List[str] = []
        for i, batch_chats in enumerate(chunked(chats_stage1, self.args.batch_size)):
            print(f"  Procesando lote {i+1}/{len(red_prompts)//self.args.batch_size + 1}")
            prompts = chats_to_prompts(self.tokenizer, batch_chats)
            completions = run_vllm_generate(self.llm, self.sampling_params, prompts)
            init_responses.extend(completions)
        
        print(f" Generadas {len(init_responses)} respuestas iniciales")
        return init_responses
    
    def stage_2_critique(self, red_prompts: List[str], init_responses: List[str]) -> tuple[List[str], List[str], List[Dict[str, str]]]:
        """
        Etapa 2: Genera críticas a las respuestas iniciales.
        
        Args:
            red_prompts: prompts originales de red teaming
            init_responses: respuestas iniciales del modelo
            
        Returns:
            Tupla con (critic_prompts, critic_responses, chosen_constitutions)
        """
        print(" Etapa 2: Generando críticas...")
        
        critic_prompts: List[str] = []
        chats_stage2: List[List[Dict[str, str]]] = []
        chosen_const_for_case: List[Dict[str, str]] = []
        
        for i, init_prompt in enumerate(red_prompts):
            # Elegir una constitución aleatoria por conversación
            c = random.choice(self.constitutions)
            chosen_const_for_case.append(c)
            
            # Construir historial de chat
            chat = build_chat_with_fewshots(self.fewshots, init_prompt)
            chat.append({"role": "assistant", "content": init_responses[i]})
            
            # Añadir prompt de crítica
            critic_prompts.append(c["critic"])
            chat.append({"role": "user", "content": c["critic"]})
            chats_stage2.append(chat)
        
        # Procesar en lotes
        critic_responses: List[str] = []
        for i, batch_chats in enumerate(chunked(chats_stage2, self.args.batch_size)):
            print(f"  Procesando lote {i+1}/{len(red_prompts)//self.args.batch_size + 1}")
            prompts = chats_to_prompts(self.tokenizer, batch_chats)
            completions = run_vllm_generate(self.llm, self.sampling_params, prompts)
            critic_responses.extend(completions)
        
        print(f" Generadas {len(critic_responses)} críticas")
        return critic_prompts, critic_responses, chosen_const_for_case
    
    def stage_3_revision(
        self, 
        red_prompts: List[str], 
        init_responses: List[str],
        critic_responses: List[str],
        chosen_const_for_case: List[Dict[str, str]]
    ) -> tuple[List[str], List[str]]:
        """
        Etapa 3: Genera respuestas revisadas basadas en las críticas.
        
        Args:
            red_prompts: prompts originales de red teaming
            init_responses: respuestas iniciales
            critic_responses: respuestas de crítica
            chosen_const_for_case: constituciones elegidas para cada caso
            
        Returns:
            Tupla con (revision_prompts, revision_responses)
        """
        print(" Etapa 3: Generando revisiones...")
        
        chats_stage3: List[List[Dict[str, str]]] = []
        revision_prompts: List[str] = []
        
        for i in range(len(red_prompts)):
            c = chosen_const_for_case[i]
            
            # Reconstruir historial completo
            chat = build_chat_with_fewshots(self.fewshots, red_prompts[i])
            chat.append({"role": "assistant", "content": init_responses[i]})
            chat.append({"role": "user", "content": c["critic"]})
            chat.append({"role": "assistant", "content": critic_responses[i]})
            
            # Añadir prompt de revisión
            revision_prompts.append(c["revision"])
            chat.append({"role": "user", "content": c["revision"]})
            chats_stage3.append(chat)
        
        # Procesar en lotes
        revision_responses: List[str] = []
        for i, batch_chats in enumerate(chunked(chats_stage3, self.args.batch_size)):
            print(f"  Procesando lote {i+1}/{len(red_prompts)//self.args.batch_size + 1}")
            prompts = chats_to_prompts(self.tokenizer, batch_chats)
            completions = run_vllm_generate(self.llm, self.sampling_params, prompts)
            revision_responses.extend(completions)
        
        print(f" Generadas {len(revision_responses)} revisiones")
        return revision_prompts, revision_responses
    
    def create_system_chat(
        self,
        red_prompts: List[str],
        init_responses: List[str],
        critic_responses: List[str],
        revision_responses: List[str],
        chosen_const_for_case: List[Dict[str, str]]
    ) -> List[List[Dict[str, str]]]:
        """
        Crea las conversaciones completas para system_chat.
        
        Returns:
            Lista de conversaciones completas con few-shots incluidos
        """
        system_chat_all_conversations: List[List[Dict[str, str]]] = []
        
        for i in range(len(red_prompts)):
            c = chosen_const_for_case[i]
            full_chat: List[Dict[str, str]] = []
            
            # Incluir few-shots
            for conv in self.fewshots:
                for msg in conv:
                    full_chat.append({"role": msg["role"], "content": msg["content"]})
            
            # Turnos principales de Constitutional AI
            full_chat.append({"role": "user", "content": red_prompts[i]})
            full_chat.append({"role": "assistant", "content": init_responses[i]})
            full_chat.append({"role": "user", "content": c["critic"]})
            full_chat.append({"role": "assistant", "content": critic_responses[i]})
            full_chat.append({"role": "user", "content": c["revision"]})
            full_chat.append({"role": "assistant", "content": revision_responses[i]})
            
            system_chat_all_conversations.append(full_chat)
        
        return system_chat_all_conversations
    
    def save_results(
        self,
        data: Dict[str, Any],
        red_prompts: List[str],
        init_responses: List[str],
        critic_prompts: List[str],
        critic_responses: List[str],
        revision_prompts: List[str],
        revision_responses: List[str],
        system_chat_conversations: List[List[Dict[str, str]]]
    ):
        """Guarda los resultados del pipeline."""
        print(" Guardando resultados...")
        
        # Crear rutas de salida
        _, out_json_path, out_parquet_path = create_output_paths(
            self.args.constitution_path, 
            self.args.output_dir
        )
        
        # Guardar constitución extendida
        save_constitution_with_results(data, system_chat_conversations, out_json_path)
        
        # Guardar dataset parquet
        save_parquet_dataset(
            red_prompts,
            init_responses,
            critic_prompts,
            critic_responses,
            revision_prompts,
            revision_responses,
            out_parquet_path
        )
        
        return out_json_path, out_parquet_path
    
    def run(self) -> tuple[str, str]:
        """
        Ejecuta el pipeline completo de Constitutional AI.
        
        Returns:
            Tupla con las rutas de los archivos de salida (json, parquet)
        """
        print(" Iniciando pipeline Constitutional AI...")
        
        # Configuración inicial
        data = self.setup()
        
        # Cargar prompts de red teaming
        red_prompts = self.load_red_teaming_prompts()
        
        # Etapa 1: Respuestas iniciales
        init_responses = self.stage_1_initial_response(red_prompts)
        
        # Etapa 2: Críticas
        critic_prompts, critic_responses, chosen_const_for_case = self.stage_2_critique(
            red_prompts, init_responses
        )
        
        # Etapa 3: Revisiones
        revision_prompts, revision_responses = self.stage_3_revision(
            red_prompts, init_responses, critic_responses, chosen_const_for_case
        )
        
        # Crear conversaciones completas
        system_chat_conversations = self.create_system_chat(
            red_prompts, init_responses, critic_responses, 
            revision_responses, chosen_const_for_case
        )
        
        # Guardar resultados
        out_json_path, out_parquet_path = self.save_results(
            data, red_prompts, init_responses, critic_prompts,
            critic_responses, revision_prompts, revision_responses,
            system_chat_conversations
        )
        
        return out_json_path, out_parquet_path