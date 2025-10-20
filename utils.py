#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funciones utilitarias generales para el pipeline Constitutional AI.
"""

import os
import random
import time
from typing import List, Any, Generator, Dict


def set_seed(seed: int) -> None:
    """Establece la semilla para reproducibilidad."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def ensure_pad_token(tokenizer) -> None:
    """Asegura que el tokenizer tenga los tokens especiales necesarios."""
    special_add = {}
    if tokenizer.pad_token is None:
        special_add["pad_token"] = "[PAD]"
    if tokenizer.sep_token is None:
        special_add["sep_token"] = ""
    if tokenizer.cls_token is None:
        special_add["cls_token"] = ""
    if tokenizer.mask_token is None:
        special_add["mask_token"] = ""
    if special_add:
        tokenizer.add_special_tokens(special_add)


def chunked(lst: List[Any], size: int) -> Generator[List[Any], None, None]:
    """Divide una lista en chunks de tamaño específico."""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def setup_vllm_environment(gpus: str) -> None:
    """Configura variables de entorno para vLLM."""
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"  # Permitir override de max_model_len
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Evitar warnings de tokenizers
    
    # Configurar GPUs visibles para vLLM
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"Configurando GPUs visibles: {gpus}")


def parse_gpu_config(gpus: str) -> tuple[int, list[str]]:
    """
    Parsea la configuración de GPUs y retorna información útil.
    
    Args:
        gpus: string con GPUs separadas por comas (ej: "0,1,2" o "1")
        
    Returns:
        Tupla con (número_de_gpus, lista_de_gpu_ids)
    """
    if not gpus or not gpus.strip():
        return 1, ["0"]  # Por defecto usar GPU 0
    
    gpu_list = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    gpu_count = len(gpu_list)
    
    print(f" Configuración de GPUs detectada: {gpu_count} GPUs -> {gpu_list}")
    return gpu_count, gpu_list


def create_output_paths(constitution_path: str, output_dir: str) -> tuple[str, str, str]:
    """
    Crea las rutas de salida para los archivos generados.
    
    Returns:
        Tupla con (directorio_experimentos, ruta_json, ruta_parquet)
    """
    constitution_dir = os.path.dirname(constitution_path) or "."
    out_dir = output_dir or constitution_dir
    exps_dir = os.path.join(out_dir, "exps")
    os.makedirs(exps_dir, exist_ok=True)
    
    base_name = os.path.basename(constitution_path).split(".")[0]
    ts = int(time.time())
    out_json_path = os.path.join(exps_dir, f"{base_name}_{ts}.json")
    out_parquet_path = os.path.join(exps_dir, f"{base_name}_{ts}.parquet")
    
    return exps_dir, out_json_path, out_parquet_path


def build_chat_with_fewshots(
    fewshots: List[List[Dict[str, str]]],
    user_prompt: str
) -> List[Dict[str, str]]:
    """
    Prepara un historial de chat con los few-shots (si existen) y el prompt del usuario.
    
    Args:
        fewshots: lista de conversaciones; cada conversación es lista de mensajes role/content.
        user_prompt: prompt del usuario actual
        
    Returns:
        Lista de mensajes con formato de chat
    """
    chat: List[Dict[str, str]] = []
    for conv in fewshots:
        for msg in conv:
            # Sanitiza mínimamente
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chat.append({"role": role, "content": content})
    # Finalmente el prompt "usuario" que dispara el paso actual
    chat.append({"role": "user", "content": user_prompt})
    return chat


def build_clean_chat(user_prompt: str) -> List[Dict[str, str]]:
    """
    Construye un chat limpio sin few-shots, solo con el prompt del usuario.
    
    Args:
        user_prompt: prompt del usuario
        
    Returns:
        Lista con un solo mensaje del usuario
    """
    return [{"role": "user", "content": user_prompt}]


def chats_to_prompts(tokenizer, chats: List[List[Dict[str, str]]]) -> List[str]:
    """
    Convierte listas de mensajes a prompts de texto usando la plantilla de chat del tokenizer.
    Se añade el "generation prompt" (prefijo del mensaje de assistant) al final.
    
    Args:
        tokenizer: tokenizer del modelo
        chats: lista de conversaciones en formato de chat
        
    Returns:
        Lista de prompts de texto formateados
    """
    prompts: List[str] = []
    for chat in chats:
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts