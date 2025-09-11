"""Core data models for GODMODE engine."""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class CognitiveMove(str, Enum):
    """Cognitive moves in progressive ladders."""
    DEFINE = "define"
    SCOPE = "scope" 
    QUANTIFY = "quantify"
    COMPARE = "compare"
    SIMULATE = "simulate"
    DECIDE = "decide"
    COMMIT = "commit"


class TriggerType(str, Enum):
    """Types of triggers for future nodes."""
    EVENT = "event"
    METRIC = "metric"
    TIME = "time"
    ANSWER_CHANGE = "answer_change"


class EntityType(str, Enum):
    """Types of entities in the ontology."""
    PERSON = "person"
    ORG = "org"
    PRODUCT = "product"
    GOAL = "goal"
    METRIC = "metric"
    PLACE = "place"
    TIME = "time"
    CONCEPT = "concept"


class ThreadStatus(str, Enum):
    """Status of conversation threads."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class Trigger(BaseModel):
    """Trigger for surfacing nodes at the right time."""
    type: TriggerType
    detail: str
    
    class Config:
        use_enum_values = True


class Question(BaseModel):
    """A question node in the ladder system."""
    id: str = Field(..., description="Unique identifier (e.g. Q001, QP2)")
    text: str = Field(..., description="The question text")
    level: int = Field(..., ge=1, description="Ladder level (must increase along builds_on chains)")
    cognitive_move: CognitiveMove = Field(..., description="Cognitive move type")
    builds_on: List[str] = Field(default_factory=list, description="IDs of parent nodes")
    delta_nuance: str = Field(..., description="New constraint/metric/frame/stake/counterfactual added")
    expected_info_gain: float = Field(..., ge=0.0, le=1.0, description="Expected information gain")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence user will want this next")
    triggers: List[Trigger] = Field(default_factory=list, description="Triggers for surfacing")
    natural_end: bool = Field(default=False, description="Whether this is a natural ending")
    tags: List[str] = Field(default_factory=list, description="Domain and intent tags")
    
    @validator('builds_on')
    def validate_builds_on(cls, v, values):
        """Validate builds_on constraints."""
        # First node in ladder can have empty builds_on
        if values.get('level') == 1:
            return v
        # Other nodes must build on at least one parent
        if not v:
            raise ValueError("Non-level-1 nodes must have at least one builds_on parent")
        return v
    
    class Config:
        use_enum_values = True


class CrossLink(BaseModel):
    """Cross-links between nodes in different lanes."""
    from_id: str = Field(..., alias="from")
    to_id: str = Field(..., alias="to") 
    type: Literal["junction"] = Field(default="junction")
    
    class Config:
        allow_population_by_field_name = True


class Lane(BaseModel):
    """A scenario lane containing a sequence of questions."""
    id: str = Field(..., description="Lane identifier (e.g. S-A, S-B)")
    name: str = Field(..., description="Lane label")
    description: str = Field(..., description="One-line lane thesis")
    lane: List[Question] = Field(..., description="Ordered questions in this lane")
    cross_links: List[CrossLink] = Field(default_factory=list, description="Cross-lane junctions")
    
    @validator('lane')
    def validate_lane_progression(cls, v):
        """Validate level progression within lane."""
        if not v:
            return v
        
        # Check level progression in builds_on chains
        for question in v:
            if question.builds_on:
                parent_levels = []
                for parent_id in question.builds_on:
                    parent = next((q for q in v if q.id == parent_id), None)
                    if parent:
                        parent_levels.append(parent.level)
                
                if parent_levels and question.level <= max(parent_levels):
                    raise ValueError(f"Question {question.id} level {question.level} must be > parent levels {parent_levels}")
        
        return v


class Thread(BaseModel):
    """A conversation thread tracking a path through the question tree."""
    thread_id: str = Field(..., description="Thread identifier")
    origin_node_id: str = Field(..., description="Starting node ID")
    path: List[str] = Field(..., description="Ordered node IDs in path")
    status: ThreadStatus = Field(..., description="Thread status")
    summary: str = Field(..., max_length=280, description="Path summary")
    
    class Config:
        use_enum_values = True


class TemporalInfo(BaseModel):
    """Temporal information for relations."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    grain: Literal["day", "week", "month", "quarter", "year"] = "day"


class Evidence(BaseModel):
    """Evidence supporting a relation."""
    source_type: Literal["user", "memory", "citation"]
    source_id: Optional[str] = None
    snippet: Optional[str] = None
    note: Optional[str] = None


class Entity(BaseModel):
    """An entity in the knowledge graph."""
    id: str = Field(..., description="Entity identifier")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in entity")
    
    class Config:
        use_enum_values = True


class Relation(BaseModel):
    """A relation between entities."""
    id: str = Field(..., description="Relation identifier")
    subj: str = Field(..., description="Subject entity ID")
    pred: str = Field(..., description="Predicate/relation type")
    obj: Union[str, Any] = Field(..., description="Object entity ID or literal value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in relation")
    hypothesis: bool = Field(default=True, description="Whether this is unverified")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    temporal: Optional[TemporalInfo] = Field(None, description="Temporal constraints")


class Mapping(BaseModel):
    """Mapping between questions and ontology elements."""
    question_id: str = Field(..., description="Question ID")
    mentions: List[str] = Field(default_factory=list, description="Entity IDs mentioned")
    claims: List[str] = Field(default_factory=list, description="Relation IDs claimed")