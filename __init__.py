"""
LLM Synthetic Reasoning Chain Generator

A clean, LLM-based system for generating high-quality multiple-choice questions 
and detailed reasoning chains from scientific text.
"""

from .llm_synthetic_reasoner import LLMSyntheticReasoner
from .llm_interface import UnifiedLLMClient, get_available_models
from .gemini_client import GeminiClient

__version__ = "1.0.0"
__all__ = [
    "LLMSyntheticReasoner",
    "UnifiedLLMClient", 
    "GeminiClient",
    "get_available_models"
]