"""
Error 0.5 AI Model Experiment Package

This package implements the scientific theory of "Error 0.5" for analyzing
LLM response consistency and uncertainty.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from .experiment import ExperimentRunner
from .metrics import MetricsCalculator
from .llm_client import LLMClient
from .utils import load_config, normalize_response, save_results

__all__ = [
    "ExperimentRunner",
    "MetricsCalculator", 
    "LLMClient",
    "load_config",
    "normalize_response",
    "save_results",
]
