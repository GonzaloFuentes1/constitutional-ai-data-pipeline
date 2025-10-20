#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestión de modelos, tokenizers y generación para el pipeline Constitutional AI.
"""

import os
from typing import List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from config import Args
from utils import ensure_pad_token, parse_gpu_config


def download_model_if_needed(model_name: str, model_path: str = "") -> str:
    """
    Descarga el modelo si no existe localmente y retorna la ruta a usar.
    
    Args:
        model_name: Nombre del modelo en HuggingFace (ej: "natong19/Mistral-Nemo-Instruct-2407-abliterated")
        model_path: Carpeta local donde descargar/buscar el modelo (vacío = usar cache por defecto)
    
    Returns:
        Ruta del modelo a usar (local o nombre de HF)
    """
    from huggingface_hub import snapshot_download
    
    # Si no se especifica model_path, usar el nombre del modelo directamente (cache por defecto)
    if not model_path:
        print(f" Usando modelo desde cache/HuggingFace: {model_name}")
        return model_name
    
    # Crear directorio si no existe
    os.makedirs(model_path, exist_ok=True)
    
    # Verificar si el modelo ya existe localmente
    model_local_path = os.path.join(model_path, model_name.replace("/", "_"))
    
    if not os.path.exists(model_local_path) or not os.path.isdir(model_local_path):
        print(f"📥 Descargando modelo {model_name} en {model_local_path}...")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_local_path,
                local_dir_use_symlinks=False
            )
            print(f" Modelo descargado exitosamente en: {model_local_path}")
        except Exception as e:
            print(f" Error descargando modelo: {e}")
            print(f" Usando modelo desde HuggingFace: {model_name}")
            return model_name
    else:
        print(f" Modelo encontrado localmente en: {model_local_path}")
    
    return model_local_path


def load_tokenizer(model_to_use: str):
    """
    Carga el tokenizer con múltiples estrategias de respaldo.
    
    Args:
        model_to_use: ruta o nombre del modelo
        
    Returns:
        Tokenizer cargado
        
    Raises:
        RuntimeError: si todas las estrategias fallan
    """
    tokenizer = None
    tokenizer_strategies = [
        {"use_fast": False, "trust_remote_code": True, "legacy": False},
        {"use_fast": True, "trust_remote_code": True, "legacy": False},
        {"use_fast": False, "trust_remote_code": True, "legacy": True},
        {"use_fast": True, "trust_remote_code": False, "legacy": False},
        {"use_fast": False, "trust_remote_code": False, "legacy": False},
    ]
    
    for i, strategy in enumerate(tokenizer_strategies, 1):
        try:
            print(f" Intentando estrategia {i}/{len(tokenizer_strategies)} para tokenizer: {strategy}")
            
            # Estrategia especial para modelos con problemas de tiktoken
            if strategy.get("legacy", False):
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_to_use,
                    trust_remote_code=strategy["trust_remote_code"],
                    use_fast=strategy["use_fast"],
                    padding_side="left",
                    add_prefix_space=False
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_to_use,
                    trust_remote_code=strategy["trust_remote_code"],
                    use_fast=strategy["use_fast"]
                )
            
            print(f" Tokenizer cargado exitosamente con estrategia {i}")
            break
            
        except Exception as e:
            print(f" Estrategia {i} falló: {e}")
            if i == len(tokenizer_strategies):
                print(f" Todas las estrategias de tokenizer fallaron")
                print(f" El modelo puede tener archivos de tokenizer corruptos o incompatibles")
                raise RuntimeError(f"Error crítico cargando tokenizer para {model_to_use}")
            continue
    
    ensure_pad_token(tokenizer)
    return tokenizer


def initialize_vllm(args: Args, model_to_use: str) -> LLM:
    """
    Inicializa el modelo vLLM.
    
    Args:
        args: configuración del pipeline
        model_to_use: ruta o nombre del modelo a cargar
        
    Returns:
        Instancia de LLM inicializada
    """
    # Calcular tensor_parallel_size automáticamente basado en GPUs disponibles
    if args.tensor_parallel_size > 0:
        tensor_parallel_size = args.tensor_parallel_size
        print(f"🔧 Usando tensor_parallel_size manual: {tensor_parallel_size}")
    else:
        # Calcular automáticamente desde las GPUs especificadas
        gpu_count, gpu_list = parse_gpu_config(args.gpus)
        tensor_parallel_size = gpu_count
        print(f"🔧 Calculando tensor_parallel_size automáticamente: {tensor_parallel_size} (basado en {gpu_count} GPUs: {gpu_list})")
    
    llm = LLM(
        model=model_to_use,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,  # por si el modelo define template especial
        enforce_eager=False,  # Usar CUDA graphs cuando sea posible
        disable_log_stats=True,  # Reducir logging verboso
    )
    print(f"✅ Modelo vLLM inicializado exitosamente con tensor_parallel_size={tensor_parallel_size}")
    return llm


def create_sampling_params(args: Args, tokenizer) -> SamplingParams:
    """
    Crea los parámetros de sampling para vLLM.
    
    Args:
        args: configuración del pipeline
        tokenizer: tokenizer del modelo
        
    Returns:
        Parámetros de sampling configurados
    """
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,  # -1 significa "disabled" en vLLM
        max_tokens=args.max_new_tokens,
        # stops razonables: eos y alguna marca común
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )


def run_vllm_generate(
    llm: LLM,
    sampling: SamplingParams,
    prompts: List[str]
) -> List[str]:
    """
    Genera respuestas para un LOTE de prompts con vLLM.
    Retorna solamente el texto de la primera hipótesis de cada prompt.
    
    Args:
        llm: instancia de vLLM
        sampling: parámetros de sampling
        prompts: lista de prompts a procesar
        
    Returns:
        Lista de respuestas generadas
    """
    outputs = llm.generate(prompts, sampling)
    # vLLM devuelve la salida en el mismo orden
    completions = []
    for out in outputs:
        if out.outputs:
            completions.append(out.outputs[0].text.strip())
        else:
            completions.append("")
    return completions