#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funciones para carga y guardado de datos del pipeline Constitutional AI.
"""

import json
import os
from typing import List, Dict, Any

import pandas as pd


def load_constitution(path: str) -> Dict[str, Any]:
    """
    Carga el archivo de constitución JSON.
    
    Args:
        path: ruta al archivo JSON de constitución
        
    Returns:
        Diccionario con los datos de constitución
        
    Raises:
        ValueError: si el archivo no tiene el formato esperado
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "constitutions" not in data or not isinstance(data["constitutions"], list):
        raise ValueError("El JSON debe contener una lista 'constitutions'.")
    if not data["constitutions"]:
        raise ValueError("La lista 'constitutions' no puede estar vacía.")

    for idx, item in enumerate(data["constitutions"]):
        if not isinstance(item, dict):
            raise ValueError(f"'constitutions[{idx}]' debe ser un objeto.")
        if "critic" not in item or "revision" not in item:
            raise ValueError(
                f"'constitutions[{idx}]' debe contener las claves 'critic' y 'revision'."
            )
        if not isinstance(item["critic"], str) or not item["critic"].strip():
            raise ValueError(f"'constitutions[{idx}].critic' debe ser string no vacío.")
        if not isinstance(item["revision"], str) or not item["revision"].strip():
            raise ValueError(f"'constitutions[{idx}].revision' debe ser string no vacío.")
    
    # few_shot_examples es opcional
    if "few_shot_examples" in data:
        if not isinstance(data["few_shot_examples"], list):
            raise ValueError("'few_shot_examples' debe ser una lista de conversaciones.")
        for conv_idx, conv in enumerate(data["few_shot_examples"]):
            if not isinstance(conv, list):
                raise ValueError(f"'few_shot_examples[{conv_idx}]' debe ser una lista de mensajes.")
            for msg_idx, msg in enumerate(conv):
                if not isinstance(msg, dict):
                    raise ValueError(
                        f"'few_shot_examples[{conv_idx}][{msg_idx}]' debe ser un objeto."
                    )
                if "role" not in msg or "content" not in msg:
                    raise ValueError(
                        f"'few_shot_examples[{conv_idx}][{msg_idx}]' debe contener 'role' y 'content'."
                    )
                if msg["role"] not in {"user", "assistant", "system"}:
                    raise ValueError(
                        f"'few_shot_examples[{conv_idx}][{msg_idx}].role' inválido: {msg['role']}"
                    )
                if not isinstance(msg["content"], str):
                    raise ValueError(
                        f"'few_shot_examples[{conv_idx}][{msg_idx}].content' debe ser string."
                    )
    else:
        data["few_shot_examples"] = []
    
    return data


def load_red_teaming_prompts_from_parquet(parquet_path: str) -> List[str]:
    """
    Carga red teaming prompts desde un archivo parquet.
    Busca los prompts en la columna 'text'.
    
    Args:
        parquet_path: Ruta al archivo parquet con los red teaming prompts
        
    Returns:
        Lista de prompts para red teaming
        
    Raises:
        ValueError: si hay problemas con el archivo o formato
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        if 'text' not in df.columns:
            raise ValueError(f"El archivo parquet debe contener una columna 'text'. Columnas encontradas: {list(df.columns)}")
        
        # Convertir a lista y filtrar valores nulos/vacíos
        prompts = df['text'].dropna().astype(str).tolist()
        prompts = [p.strip() for p in prompts if p.strip()]
        
        print(f"Cargados {len(prompts)} red teaming prompts desde: {parquet_path}")
        return prompts
        
    except Exception as e:
        raise ValueError(f"Error cargando red teaming prompts desde {parquet_path}: {str(e)}")


def save_constitution_with_results(
    data: Dict[str, Any],
    system_chat_conversations: List[List[Dict[str, str]]],
    output_path: str
) -> None:
    """
    Guarda la constitución extendida con los resultados del pipeline.
    
    Args:
        data: datos originales de la constitución
        system_chat_conversations: conversaciones generadas por el pipeline
        output_path: ruta donde guardar el archivo JSON
    """
    data_out = dict(data)  # copia para no mutar el original
    data_out["system_chat"] = system_chat_conversations
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)
    
    print(f"Guardado JSON extendido en: {output_path}")


def save_parquet_dataset(
    red_prompts: List[str],
    init_responses: List[str],
    critic_prompts: List[str],
    critic_responses: List[str], 
    revision_prompts: List[str],
    revision_responses: List[str],
    output_path: str
) -> None:
    """
    Guarda el dataset en formato parquet con TODAS las columnas especificadas:
    - init_prompt, init_response, critic_prompt, critic_response, revision_prompt, revision_response
    - prompt: el prompt inicial del red-teaming  
    - messages: lista de mensajes de TODO el proceso Constitutional AI (6 pasos) SIN few-shots
    - chosen: la respuesta revisada (final)
    - rejected: la respuesta inicial (antes de la revisión)
    
    Args:
        red_prompts: prompts de red teaming iniciales
        init_responses: respuestas iniciales del modelo
        critic_prompts: prompts de crítica
        critic_responses: respuestas de crítica
        revision_prompts: prompts de revisión
        revision_responses: respuestas revisadas finales
        output_path: ruta donde guardar el archivo parquet
    """
    data_rows = []
    
    for i in range(len(red_prompts)):
        # Construir la conversación completa del proceso Constitutional AI (sin few-shots)
        messages = [
            {"role": "user", "content": red_prompts[i]},           # 1. Prompt red-teaming
            {"role": "assistant", "content": init_responses[i]},   # 2. Respuesta inicial
            {"role": "user", "content": critic_prompts[i]},       # 3. Prompt de crítica
            {"role": "assistant", "content": critic_responses[i]}, # 4. Respuesta de crítica
            {"role": "user", "content": revision_prompts[i]},     # 5. Prompt de revisión
            {"role": "assistant", "content": revision_responses[i]} # 6. Respuesta final revisada
        ]
        
        row = {
            # Columnas individuales de cada paso
            "init_prompt": red_prompts[i],
            "init_response": init_responses[i],
            "critic_prompt": critic_prompts[i], 
            "critic_response": critic_responses[i],
            "revision_prompt": revision_prompts[i],
            "revision_response": revision_responses[i],
            # Columnas del formato de entrenamiento
            "prompt": red_prompts[i],
            "messages": messages,
            "chosen": revision_responses[i],  # respuesta revisada/mejorada
            "rejected": init_responses[i]     # respuesta inicial
        }
        data_rows.append(row)
    
    # Crear DataFrame y guardar como parquet
    df = pd.DataFrame(data_rows)
    df.to_parquet(output_path, index=False)
    print(f"Guardado dataset parquet en: {output_path}")


def load_base_prompt(prompt_path: str) -> str:
    """
    Carga un base prompt desde archivo.
    
    Args:
        prompt_path: ruta al archivo de prompt
        
    Returns:
        Contenido del prompt como string
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()