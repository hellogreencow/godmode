"""
Core data models for the GodMode hierarchical reasoning system.

This module defines the fundamental data structures used throughout the system,
implementing advanced type safety and validation using Pydantic v2.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from uuid import UUID

import numpy as np
import torch
from pydantic import BaseModel, Field, ConfigDict, validator, computed_field
from pydantic.functional_validators import field_validator


T = TypeVar('T')


class ReasoningType(str, Enum):
    """Types of reasoning processes supported by the system."""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"
    HIERARCHICAL = "hierarchical"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"


class CognitiveLevel(str, Enum):
    """Levels in the hierarchical cognitive architecture."""
    METACOGNITIVE = "metacognitive"      # Strategic planning and meta-reasoning
    EXECUTIVE = "executive"              # Goal management and control
    OPERATIONAL = "operational"          # Task execution and procedures
    REACTIVE = "reactive"                # Immediate responses and reflexes


class AttentionMechanism(str, Enum):
    """Types of attention mechanisms."""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD = "multi_head"
    HIERARCHICAL = "hierarchical"
    SPARSE = "sparse"
    ADAPTIVE = "adaptive"


class BaseGodModeModel(BaseModel):
    """Base model with advanced configuration for all GodMode models."""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        
        # Serialization
        ser_json_inf_nan='constants',
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            np.ndarray: lambda v: v.tolist(),
            torch.Tensor: lambda v: v.detach().cpu().numpy().tolist(),
        },
        
        # Validation
        str_strip_whitespace=True,
        validate_default=True,
    )
    
    # Metadata
    id: UUID = Field(default_factory=uuid.uuid4, description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    def __hash__(self) -> int:
        """Enable hashing for use in sets and as dict keys."""
        return hash(self.id)


class TensorData(BaseModel):
    """Wrapper for tensor data with metadata."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data: Union[torch.Tensor, np.ndarray, List[float]]
    shape: List[int]
    dtype: str
    device: Optional[str] = None
    requires_grad: bool = False
    
    @field_validator('data')
    @classmethod
    def validate_tensor_data(cls, v):
        """Validate and normalize tensor data."""
        if isinstance(v, torch.Tensor):
            return v
        elif isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        elif isinstance(v, (list, tuple)):
            return torch.tensor(v)
        else:
            raise ValueError(f"Unsupported tensor data type: {type(v)}")
    
    @computed_field
    @property
    def size(self) -> int:
        """Total number of elements in the tensor."""
        return int(np.prod(self.shape))


class AttentionWeights(BaseGodModeModel):
    """Attention weights for hierarchical reasoning."""
    
    mechanism: AttentionMechanism
    weights: TensorData
    query_dim: int
    key_dim: int
    value_dim: int
    num_heads: int = 1
    temperature: float = 1.0
    dropout_rate: float = 0.1
    
    # Hierarchical attention specific
    level_weights: Optional[Dict[CognitiveLevel, float]] = None
    cross_level_attention: Optional[TensorData] = None
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Ensure temperature is positive."""
        if v <= 0:
            raise ValueError("Temperature must be positive")
        return v


class KnowledgeNode(BaseGodModeModel):
    """Node in the knowledge graph representing a concept or entity."""
    
    name: str = Field(..., min_length=1, description="Node name/identifier")
    node_type: str = Field(..., description="Type of knowledge node")
    embedding: Optional[TensorData] = Field(None, description="Vector embedding")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    
    # Semantic information
    ontology_class: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)
    
    # Temporal information
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    def is_valid_at(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if the node is valid at a given timestamp."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True


class GraphRelation(BaseGodModeModel):
    """Relation between knowledge nodes in the graph."""
    
    subject: UUID = Field(..., description="Subject node ID")
    predicate: str = Field(..., min_length=1, description="Relation type")
    object: UUID = Field(..., description="Object node ID")
    
    # Relation properties
    weight: float = Field(1.0, ge=0.0, description="Relation strength")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in relation")
    bidirectional: bool = Field(False, description="Whether relation is bidirectional")
    
    # Temporal information
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Provenance
    source: Optional[str] = Field(None, description="Source of the relation")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")


class ReasoningStep(BaseGodModeModel):
    """Individual step in a reasoning process."""
    
    step_number: int = Field(..., ge=0, description="Step order in the reasoning process")
    operation: str = Field(..., description="Type of reasoning operation")
    input_state: Dict[str, Any] = Field(default_factory=dict, description="Input state")
    output_state: Dict[str, Any] = Field(default_factory=dict, description="Output state")
    
    # Cognitive information
    cognitive_level: CognitiveLevel
    attention_weights: Optional[AttentionWeights] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Step confidence")
    
    # Performance metrics
    execution_time: Optional[float] = Field(None, ge=0.0, description="Execution time in seconds")
    memory_usage: Optional[int] = Field(None, ge=0, description="Memory usage in bytes")
    
    # Explanations
    rationale: Optional[str] = Field(None, description="Human-readable explanation")
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative steps considered")


class MemoryState(BaseGodModeModel):
    """State of the cognitive memory system."""
    
    working_memory: Dict[str, Any] = Field(default_factory=dict, description="Working memory contents")
    long_term_memory: Dict[str, Any] = Field(default_factory=dict, description="Long-term memory references")
    episodic_memory: List[Dict[str, Any]] = Field(default_factory=list, description="Episodic memories")
    
    # Memory metrics
    working_memory_capacity: int = Field(7, ge=1, le=20, description="Working memory capacity")
    memory_utilization: float = Field(0.0, ge=0.0, le=1.0, description="Current memory usage")
    
    # Attention and focus
    current_focus: Optional[str] = Field(None, description="Current focus of attention")
    attention_breadth: float = Field(0.5, ge=0.0, le=1.0, description="Breadth of attention")
    
    def get_memory_pressure(self) -> float:
        """Calculate current memory pressure."""
        return self.memory_utilization * (1.0 + max(0, len(self.working_memory) - self.working_memory_capacity))


class HierarchicalContext(BaseGodModeModel):
    """Context information for hierarchical reasoning."""
    
    current_level: CognitiveLevel
    parent_context: Optional[UUID] = Field(None, description="Parent context ID")
    child_contexts: List[UUID] = Field(default_factory=list, description="Child context IDs")
    
    # Level-specific information
    goals: List[str] = Field(default_factory=list, description="Goals at this level")
    constraints: List[str] = Field(default_factory=list, description="Constraints at this level")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Available resources")
    
    # Cross-level communication
    upward_messages: List[Dict[str, Any]] = Field(default_factory=list, description="Messages to parent level")
    downward_messages: List[Dict[str, Any]] = Field(default_factory=list, description="Messages to child levels")
    
    # Temporal information
    time_horizon: Optional[float] = Field(None, ge=0.0, description="Planning time horizon")
    deadline: Optional[datetime] = Field(None, description="Context deadline")
    
    def is_root_context(self) -> bool:
        """Check if this is a root context."""
        return self.parent_context is None
    
    def is_leaf_context(self) -> bool:
        """Check if this is a leaf context."""
        return len(self.child_contexts) == 0


class CognitiveState(BaseGodModeModel):
    """Complete cognitive state of the reasoning system."""
    
    memory_state: MemoryState
    hierarchical_context: HierarchicalContext
    current_reasoning_type: ReasoningType
    
    # Active processes
    active_goals: List[str] = Field(default_factory=list, description="Currently active goals")
    suspended_goals: List[str] = Field(default_factory=list, description="Suspended goals")
    completed_goals: List[str] = Field(default_factory=list, description="Completed goals")
    
    # Emotional and motivational state
    arousal_level: float = Field(0.5, ge=0.0, le=1.0, description="Cognitive arousal level")
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence")
    motivation: float = Field(0.5, ge=0.0, le=1.0, description="Motivation level")
    
    # Meta-cognitive information
    confidence_in_state: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in current state")
    uncertainty: float = Field(0.5, ge=0.0, le=1.0, description="Uncertainty level")
    
    def get_cognitive_load(self) -> float:
        """Calculate current cognitive load."""
        memory_load = self.memory_state.get_memory_pressure()
        goal_load = len(self.active_goals) / 10.0  # Normalize by typical max goals
        arousal_factor = abs(self.arousal_level - 0.5) * 2  # Optimal arousal is 0.5
        
        return min(1.0, (memory_load + goal_load + arousal_factor) / 3.0)


class Problem(BaseGodModeModel):
    """Representation of a problem to be solved."""
    
    title: str = Field(..., min_length=1, description="Problem title")
    description: str = Field(..., min_length=1, description="Detailed problem description")
    problem_type: str = Field(..., description="Type/category of the problem")
    
    # Problem structure
    domain: str = Field(..., description="Problem domain")
    complexity: str = Field("medium", description="Problem complexity level")
    constraints: List[str] = Field(default_factory=list, description="Problem constraints")
    objectives: List[str] = Field(default_factory=list, description="Problem objectives")
    
    # Context and background
    background_knowledge: Dict[str, Any] = Field(default_factory=dict, description="Relevant background")
    related_problems: List[UUID] = Field(default_factory=list, description="Related problem IDs")
    
    # Requirements
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")
    time_limit: Optional[float] = Field(None, ge=0.0, description="Time limit in seconds")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource constraints")
    
    # Metadata
    priority: int = Field(0, description="Problem priority")
    tags: List[str] = Field(default_factory=list, description="Problem tags")


class Solution(BaseGodModeModel):
    """Solution to a problem with detailed reasoning trace."""
    
    problem_id: UUID = Field(..., description="ID of the solved problem")
    solution_text: str = Field(..., description="Human-readable solution")
    solution_data: Dict[str, Any] = Field(default_factory=dict, description="Structured solution data")
    
    # Quality metrics
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Solution confidence")
    completeness: float = Field(0.0, ge=0.0, le=1.0, description="Solution completeness")
    feasibility: float = Field(0.0, ge=0.0, le=1.0, description="Solution feasibility")
    
    # Performance metrics
    solving_time: Optional[float] = Field(None, ge=0.0, description="Time to solve in seconds")
    reasoning_steps: int = Field(0, ge=0, description="Number of reasoning steps")
    memory_peak: Optional[int] = Field(None, ge=0, description="Peak memory usage")
    
    # Validation
    is_validated: bool = Field(False, description="Whether solution is validated")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
    
    # Alternative solutions
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative solutions")
    
    def get_overall_quality(self) -> float:
        """Calculate overall solution quality score."""
        return (self.confidence + self.completeness + self.feasibility) / 3.0


class ReasoningTrace(BaseGodModeModel):
    """Complete trace of a reasoning process."""
    
    problem_id: UUID = Field(..., description="ID of the problem being solved")
    solution_id: Optional[UUID] = Field(None, description="ID of the resulting solution")
    
    # Reasoning process
    reasoning_type: ReasoningType
    steps: List[ReasoningStep] = Field(default_factory=list, description="Reasoning steps")
    cognitive_states: List[CognitiveState] = Field(default_factory=list, description="Cognitive states")
    
    # Process metrics
    total_time: Optional[float] = Field(None, ge=0.0, description="Total reasoning time")
    convergence_steps: Optional[int] = Field(None, ge=0, description="Steps to convergence")
    
    # Quality metrics
    coherence: float = Field(0.0, ge=0.0, le=1.0, description="Reasoning coherence")
    consistency: float = Field(0.0, ge=0.0, le=1.0, description="Reasoning consistency")
    efficiency: float = Field(0.0, ge=0.0, le=1.0, description="Reasoning efficiency")
    
    # Hierarchical information
    hierarchical_levels: Dict[CognitiveLevel, List[int]] = Field(
        default_factory=dict, 
        description="Steps at each cognitive level"
    )
    
    # Analysis
    bottlenecks: List[Dict[str, Any]] = Field(default_factory=list, description="Identified bottlenecks")
    insights: List[str] = Field(default_factory=list, description="Insights gained")
    
    def get_step_at_level(self, level: CognitiveLevel) -> List[ReasoningStep]:
        """Get all reasoning steps at a specific cognitive level."""
        if level not in self.hierarchical_levels:
            return []
        
        step_indices = self.hierarchical_levels[level]
        return [self.steps[i] for i in step_indices if i < len(self.steps)]
    
    def get_reasoning_efficiency(self) -> float:
        """Calculate reasoning efficiency based on steps and time."""
        if not self.steps or not self.total_time:
            return 0.0
        
        # Efficiency is inversely related to time and steps, but positively to quality
        time_efficiency = 1.0 / (1.0 + self.total_time / 60.0)  # Normalize by minute
        step_efficiency = 1.0 / (1.0 + len(self.steps) / 10.0)  # Normalize by 10 steps
        quality_factor = (self.coherence + self.consistency) / 2.0
        
        return (time_efficiency + step_efficiency + quality_factor) / 3.0
