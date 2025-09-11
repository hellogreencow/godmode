"""
External API integrations for GodMode system.
"""

from .openrouter import OpenRouterIntegration
from .model_selector import ModelSelector

__all__ = [
    "OpenRouterIntegration",
    "ModelSelector",
]