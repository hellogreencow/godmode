"""
GODMODE - Superhuman Question Foresight Engine

See the questions before you ask them.
"""

__version__ = "0.1.0"
__author__ = "GODMODE Team"

from .core.engine import GodmodeEngine
from .models.core import Question, Lane, Thread, Entity, Relation
from .models.commands import Command, InitCommand, AdvanceCommand

__all__ = [
    "GodmodeEngine",
    "Question",
    "Lane", 
    "Thread",
    "Entity",
    "Relation",
    "Command",
    "InitCommand",
    "AdvanceCommand",
]