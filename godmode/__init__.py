"""
GodMode: Advanced Hierarchical Reasoning System

A cutting-edge AI reasoning system implementing hierarchical cognitive architectures
for complex problem-solving with neural networks, knowledge graphs, and multi-level reasoning.
"""

__version__ = "1.0.0"
__author__ = "GodMode AI"
__email__ = "ai@godmode.dev"
__license__ = "MIT"

# Core imports
from godmode.core.engine import GodModeEngine
from godmode.core.memory import CognitiveMemory, WorkingMemory
from godmode.models.core import (
    Problem,
    Solution,
    ReasoningTrace,
    CognitiveState,
    HierarchicalContext,
)

# Reasoning models
from godmode.core.reasoning.forward import ForwardReasoningEngine
from godmode.core.reasoning.backward import BackwardReasoningEngine
from godmode.core.reasoning.cognitive_moves import CognitiveMoveEngine

# Experimental features
from godmode.experimental.hierarchical_reasoning import (
    HierarchicalReasoningModel,
    MultiLevelAttention,
    CognitiveArchitecture,
)

# Web interface
from godmode.web.app import create_app

__all__ = [
    # Core
    "GodModeEngine",
    "CognitiveMemory",
    "WorkingMemory",
    
    # Models
    "Problem",
    "Solution", 
    "ReasoningTrace",
    "CognitiveState",
    "HierarchicalContext",
    
    # Reasoning
    "ForwardReasoningEngine",
    "BackwardReasoningEngine", 
    "CognitiveMoveEngine",
    
    # Experimental
    "HierarchicalReasoningModel",
    "MultiLevelAttention",
    "CognitiveArchitecture",
    
    # Web
    "create_app",
]

# Package metadata
__all__.extend(["__version__", "__author__", "__email__", "__license__"])
