"""Command models for GODMODE interface."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class Budgets(BaseModel):
    """Budget constraints for processing."""
    beam_width: int = Field(default=4, description="Candidates per layer")
    depth_back: int = Field(default=4, description="Priors ladder depth per pass")
    depth_fwd: int = Field(default=5, description="Futures ladder depth per pass")
    max_tokens_reply: int = Field(default=160, description="Chat reply token limit")
    time_s: float = Field(default=2.5, description="Soft time target per update")
    prune_if_info_gain_below: float = Field(default=0.18, description="Prune threshold")


class Command(BaseModel, ABC):
    """Base command interface."""
    
    @abstractmethod
    def get_command_type(self) -> str:
        """Get the command type identifier."""
        pass


class InitCommand(Command):
    """Initialize GODMODE with a current question."""
    current_question: str = Field(..., description="The question to analyze")
    context: Optional[str] = Field(None, description="Optional context")
    budgets: Optional[Budgets] = Field(None, description="Custom budget overrides")
    
    def get_command_type(self) -> str:
        return "INIT"


class AdvanceCommand(Command):
    """Advance/expand around a chosen node."""
    node_id: str = Field(..., description="Node ID to expand around")
    user_answer: Optional[str] = Field(None, description="Optional inline answer")
    
    def get_command_type(self) -> str:
        return "ADVANCE"


class ContinueCommand(Command):
    """Continue the deepest promising lane."""
    thread_id: str = Field(..., description="Thread ID to continue")
    
    def get_command_type(self) -> str:
        return "CONTINUE"


class SummarizeCommand(Command):
    """Get path summary and recommended next step."""
    thread_id: str = Field(..., description="Thread ID to summarize")
    
    def get_command_type(self) -> str:
        return "SUMMARIZE"


class RegraftCommand(Command):
    """Move a sub-branch to a different lane."""
    from_node_id: str = Field(..., description="Source node ID")
    to_lane_id: str = Field(..., description="Target lane ID")
    
    def get_command_type(self) -> str:
        return "REGRAFT"


class MergeCommand(Command):
    """Merge concurrent branches into unified path."""
    thread_ids: List[str] = Field(..., description="Thread IDs to merge")
    
    def get_command_type(self) -> str:
        return "MERGE"