#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constitutional AI batched pipeline (vLLM version, no llm-swarm) - Versión modularizada.

Script principal que coordina todos los módulos del pipeline Constitutional AI.

- Modelo por defecto: natong19/Mistral-Nemo-Instruct-2407-abliterated
- Carga few-shot-examples desde el mismo JSON de la constitución.
- Pipeline en 3 pasos: red-teaming -> crítica -> revisión.
- Batching real con vLLM: se generan muchos prompts en paralelo por etapa.

Salida:
- Copia extendida de la constitución en exps/<nombre>_<ts>.json (con `system_chat`).
- Dataset en exps/<nombre>_<ts>.jsonl con campos:
  {init_prompt, init_response, critic_prompt, critic_response, revision_prompt, revision_response}

Uso:
    python main.py --constitution_path constitution_anthropic.json --max_samples 128

Módulos:
    - config.py: Configuración y parámetros
    - utils.py: Funciones utilitarias generales
    - data_handler.py: Carga y guardado de datos
    - model_manager.py: Gestión de modelos y tokenizers
    - pipeline.py: Lógica principal del pipeline
"""

from config import parse_args
from pipeline import ConstitutionalAIPipeline


def main():
    """Función principal que ejecuta el pipeline Constitutional AI."""
    try:
        # Parsear argumentos de configuración
        args = parse_args()
        
        # Crear y ejecutar pipeline
        pipeline = ConstitutionalAIPipeline(args)
        out_json_path, out_parquet_path = pipeline.run()
        
        # Mostrar resumen final
        print("\n" + "="*60)
        print(" PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f" Constitución extendida: {out_json_path}")
        print(f" Dataset parquet: {out_parquet_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nPipeline interrumpido por el usuario")
    except Exception as e:
        print(f"\n Error ejecutando pipeline: {e}")
        raise


if __name__ == "__main__":
    main()