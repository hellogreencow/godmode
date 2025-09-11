"""
Command models for the GodMode system.

This module defines command structures for various system operations,
implementing the Command pattern for extensible and traceable operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import Field

from godmode.models.core import BaseGodModeModel, ReasoningType, CognitiveLevel


class CommandType(str, Enum):
    """Types of commands supported by the system."""
    REASONING = "reasoning"
    MEMORY = "memory"
    QUERY = "query"
    UPDATE = "update"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    EXPORT = "export"


class CommandPriority(str, Enum):
    """Command execution priorities."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CommandStatus(str, Enum):
    """Command execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Command(BaseGodModeModel, ABC):
    """Base class for all system commands."""
    
    command_type: CommandType
    priority: CommandPriority = CommandPriority.NORMAL
    status: CommandStatus = CommandStatus.PENDING
    
    # Execution context
    user_id: Optional[str] = Field(None, description="User who issued the command")
    session_id: Optional[UUID] = Field(None, description="Session context")
    parent_command_id: Optional[UUID] = Field(None, description="Parent command ID")
    
    # Timing
    scheduled_at: Optional[datetime] = Field(None, description="When to execute the command")
    started_at: Optional[datetime] = Field(None, description="When execution started")
    completed_at: Optional[datetime] = Field(None, description="When execution completed")
    timeout: Optional[float] = Field(None, ge=0.0, description="Timeout in seconds")
    
    # Results and error handling
    result: Optional[Dict[str, Any]] = Field(None, description="Command result")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Command tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate command parameters."""
        pass
    
    @abstractmethod
    def get_estimated_duration(self) -> float:
        """Get estimated execution duration in seconds."""
        pass
    
    def is_executable(self) -> bool:
        """Check if command is ready for execution."""
        return (
            self.status == CommandStatus.PENDING and
            self.validate_parameters() and
            (self.scheduled_at is None or self.scheduled_at <= datetime.utcnow())
        )
    
    def get_execution_time(self) -> Optional[float]:
        """Get actual execution time if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ReasoningCommand(Command):
    """Command for reasoning operations."""
    
    command_type: CommandType = CommandType.REASONING
    
    # Problem specification
    problem_id: Optional[UUID] = Field(None, description="Existing problem ID")
    problem_description: Optional[str] = Field(None, description="New problem description")
    problem_type: Optional[str] = Field(None, description="Problem type")
    domain: Optional[str] = Field(None, description="Problem domain")
    
    # Reasoning configuration
    reasoning_type: ReasoningType = ReasoningType.HIERARCHICAL
    max_steps: int = Field(100, ge=1, le=10000, description="Maximum reasoning steps")
    max_time: Optional[float] = Field(None, ge=0.0, description="Maximum reasoning time")
    
    # Hierarchical configuration
    cognitive_levels: List[CognitiveLevel] = Field(
        default_factory=lambda: list(CognitiveLevel),
        description="Cognitive levels to use"
    )
    start_level: CognitiveLevel = CognitiveLevel.METACOGNITIVE
    
    # Quality requirements
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum solution confidence")
    require_explanation: bool = Field(True, description="Whether to generate explanations")
    
    # Advanced options
    use_memory: bool = Field(True, description="Whether to use memory system")
    use_knowledge_graph: bool = Field(True, description="Whether to use knowledge graph")
    parallel_branches: int = Field(1, ge=1, le=10, description="Number of parallel reasoning branches")
    
    def validate_parameters(self) -> bool:
        """Validate reasoning command parameters."""
        if not self.problem_id and not self.problem_description:
            return False
        
        if self.max_time and self.max_time <= 0:
            return False
            
        if not self.cognitive_levels:
            return False
            
        return True
    
    def get_estimated_duration(self) -> float:
        """Estimate reasoning duration based on complexity."""
        base_time = 1.0  # Base time in seconds
        
        # Adjust for reasoning type
        type_multipliers = {
            ReasoningType.FORWARD: 1.0,
            ReasoningType.BACKWARD: 1.2,
            ReasoningType.BIDIRECTIONAL: 1.5,
            ReasoningType.HIERARCHICAL: 2.0,
            ReasoningType.ABDUCTIVE: 1.8,
            ReasoningType.ANALOGICAL: 1.3,
            ReasoningType.CAUSAL: 1.6,
            ReasoningType.TEMPORAL: 1.4,
        }
        
        time_estimate = base_time * type_multipliers.get(self.reasoning_type, 1.0)
        
        # Adjust for steps and levels
        time_estimate *= (self.max_steps / 100.0)
        time_estimate *= len(self.cognitive_levels)
        time_estimate *= self.parallel_branches
        
        # Apply configured maximum
        if self.max_time:
            time_estimate = min(time_estimate, self.max_time)
            
        return time_estimate


class MemoryCommand(Command):
    """Command for memory operations."""
    
    command_type: CommandType = CommandType.MEMORY
    
    # Operation type
    operation: str = Field(..., description="Memory operation type")
    
    # Memory targets
    working_memory: bool = Field(False, description="Target working memory")
    long_term_memory: bool = Field(False, description="Target long-term memory")
    episodic_memory: bool = Field(False, description="Target episodic memory")
    
    # Data
    memory_key: Optional[str] = Field(None, description="Memory key for operations")
    memory_data: Optional[Dict[str, Any]] = Field(None, description="Memory data")
    query_pattern: Optional[str] = Field(None, description="Query pattern for retrieval")
    
    # Configuration
    max_results: int = Field(100, ge=1, le=10000, description="Maximum results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    
    def validate_parameters(self) -> bool:
        """Validate memory command parameters."""
        if not any([self.working_memory, self.long_term_memory, self.episodic_memory]):
            return False
            
        valid_operations = [
            "store", "retrieve", "update", "delete", "search", "consolidate", "forget"
        ]
        if self.operation not in valid_operations:
            return False
            
        return True
    
    def get_estimated_duration(self) -> float:
        """Estimate memory operation duration."""
        base_times = {
            "store": 0.1,
            "retrieve": 0.2,
            "update": 0.15,
            "delete": 0.05,
            "search": 0.5,
            "consolidate": 2.0,
            "forget": 0.3,
        }
        
        base_time = base_times.get(self.operation, 0.2)
        
        # Adjust for memory types
        memory_multiplier = 1.0
        if self.long_term_memory:
            memory_multiplier += 0.5
        if self.episodic_memory:
            memory_multiplier += 0.3
            
        return base_time * memory_multiplier


class QueryCommand(Command):
    """Command for querying the system."""
    
    command_type: CommandType = CommandType.QUERY
    
    # Query specification
    query_text: str = Field(..., min_length=1, description="Query text")
    query_type: str = Field("semantic", description="Type of query")
    
    # Target systems
    knowledge_graph: bool = Field(True, description="Query knowledge graph")
    memory_system: bool = Field(True, description="Query memory system")
    reasoning_traces: bool = Field(False, description="Query reasoning traces")
    
    # Configuration
    max_results: int = Field(10, ge=1, le=1000, description="Maximum results")
    include_confidence: bool = Field(True, description="Include confidence scores")
    include_explanations: bool = Field(False, description="Include explanations")
    
    # Filtering
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")
    
    def validate_parameters(self) -> bool:
        """Validate query command parameters."""
        if not self.query_text.strip():
            return False
            
        if not any([self.knowledge_graph, self.memory_system, self.reasoning_traces]):
            return False
            
        return True
    
    def get_estimated_duration(self) -> float:
        """Estimate query duration."""
        base_time = 0.5
        
        # Adjust for query complexity
        complexity_multiplier = len(self.query_text.split()) / 10.0
        
        # Adjust for target systems
        system_multiplier = 0.0
        if self.knowledge_graph:
            system_multiplier += 1.0
        if self.memory_system:
            system_multiplier += 0.5
        if self.reasoning_traces:
            system_multiplier += 0.3
            
        return base_time * max(1.0, complexity_multiplier) * max(1.0, system_multiplier)


class UpdateCommand(Command):
    """Command for updating system components."""
    
    command_type: CommandType = CommandType.UPDATE
    
    # Update target
    target_type: str = Field(..., description="Type of component to update")
    target_id: UUID = Field(..., description="ID of component to update")
    
    # Update data
    updates: Dict[str, Any] = Field(..., description="Update data")
    merge_strategy: str = Field("merge", description="How to merge updates")
    
    # Validation
    validate_before_update: bool = Field(True, description="Validate before updating")
    backup_before_update: bool = Field(True, description="Create backup before updating")
    
    # Constraints
    allowed_fields: Optional[List[str]] = Field(None, description="Fields allowed to update")
    forbidden_fields: List[str] = Field(default_factory=list, description="Fields forbidden to update")
    
    def validate_parameters(self) -> bool:
        """Validate update command parameters."""
        if not self.updates:
            return False
            
        valid_targets = [
            "problem", "solution", "memory", "knowledge_node", "relation", "configuration"
        ]
        if self.target_type not in valid_targets:
            return False
            
        valid_strategies = ["merge", "replace", "patch"]
        if self.merge_strategy not in valid_strategies:
            return False
            
        return True
    
    def get_estimated_duration(self) -> float:
        """Estimate update duration."""
        base_time = 0.2
        
        # Adjust for update size
        update_multiplier = len(self.updates) / 10.0
        
        # Adjust for validation and backup
        if self.validate_before_update:
            base_time += 0.1
        if self.backup_before_update:
            base_time += 0.05
            
        return base_time * max(1.0, update_multiplier)


class BatchCommand(Command):
    """Command for executing multiple commands as a batch."""
    
    command_type: CommandType = CommandType.UPDATE  # Can be any type
    
    # Batch configuration
    commands: List[Command] = Field(..., min_length=1, description="Commands to execute")
    execution_mode: str = Field("sequential", description="Execution mode")
    fail_fast: bool = Field(False, description="Stop on first failure")
    
    # Dependencies
    dependency_graph: Dict[UUID, List[UUID]] = Field(
        default_factory=dict,
        description="Command dependencies"
    )
    
    def validate_parameters(self) -> bool:
        """Validate batch command parameters."""
        if not self.commands:
            return False
            
        # Validate individual commands
        for cmd in self.commands:
            if not cmd.validate_parameters():
                return False
                
        # Validate execution mode
        valid_modes = ["sequential", "parallel", "dependency_order"]
        if self.execution_mode not in valid_modes:
            return False
            
        return True
    
    def get_estimated_duration(self) -> float:
        """Estimate batch execution duration."""
        if self.execution_mode == "parallel":
            # Parallel execution: max of individual durations
            return max(cmd.get_estimated_duration() for cmd in self.commands)
        else:
            # Sequential execution: sum of individual durations
            return sum(cmd.get_estimated_duration() for cmd in self.commands)
