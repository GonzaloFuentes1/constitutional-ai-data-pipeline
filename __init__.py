#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constitutional AI Data Pipeline - Versión modularizada.

Este paquete implementa un pipeline de Constitutional AI usando vLLM para
procesamiento en lotes eficiente.

Módulos:
    config: Configuración y parámetros del pipeline
    utils: Funciones utilitarias generales
    data_handler: Carga y guardado de datos
    model_manager: Gestión de modelos y tokenizers
    pipeline: Lógica principal del pipeline Constitutional AI
    main: Script principal coordinador

Ejemplo de uso:
    python main.py --constitution_path constitution_anthropic.json --max_samples 128
"""

__version__ = "1.0.0"
__author__ = "Gonzalo Fuentes"

from .config import Args, parse_args
from .pipeline import ConstitutionalAIPipeline

__all__ = ["Args", "parse_args", "ConstitutionalAIPipeline"]