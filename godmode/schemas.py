"""
Strict JSON schemas for GODMODE outputs
"""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime


class Trigger(BaseModel):
    type: Literal["event", "metric", "time", "answer_change"]
    detail: str


class TemporalInfo(BaseModel):
    start: Optional[str] = None  # ISO format
    end: Optional[str] = None    # ISO format
    grain: Literal["day", "week", "month", "quarter", "year"]


class Evidence(BaseModel):
    source_type: Literal["user", "memory", "citation"]
    source_id: Optional[str] = None
    snippet: Optional[str] = None
    note: Optional[str] = None


class QuestionNode(BaseModel):
    id: str = Field(..., pattern=r"^Q\d+$")
    text: str
    level: int = Field(..., ge=1)
    cognitive_move: Literal["define", "scope", "quantify", "compare", "simulate", "decide", "commit"]
    builds_on: List[str] = Field(default_factory=list)
    delta_nuance: str
    expected_info_gain: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    triggers: List[Trigger] = Field(default_factory=list)
    natural_end: bool = False
    tags: List[str] = Field(default_factory=list)

    @validator('builds_on')
    def validate_builds_on(cls, v, values):
        if 'level' in values and values['level'] > 1 and not v:
            raise ValueError("Nodes at level > 1 must build on at least one earlier node")
        return v


class CrossLink(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    type: Literal["junction"]

    class Config:
        allow_population_by_field_name = True


class ScenarioLane(BaseModel):
    id: str = Field(..., pattern=r"^S-[A-Z]$")
    name: str
    description: str  # one-line lane thesis
    lane: List[QuestionNode]
    cross_links: List[CrossLink] = Field(default_factory=list)


class Thread(BaseModel):
    thread_id: str = Field(..., pattern=r"^T\d+$")
    origin_node_id: str
    path: List[str]  # ordered node IDs
    status: Literal["active", "paused", "ended"]
    summary: str = Field(..., max_length=280)


class BudgetsUsed(BaseModel):
    beam_width: int = 4
    depth_back: int = 4
    depth_fwd: int = 5
    time_s: float = 0.0


class GraphMeta(BaseModel):
    version: str = "1.0"
    budgets_used: BudgetsUsed
    notes: Optional[str] = None


class GraphUpdate(BaseModel):
    current_question: str
    priors: List[QuestionNode]
    scenarios: List[ScenarioLane]
    threads: List[Thread]
    meta: GraphMeta


class Entity(BaseModel):
    id: str = Field(..., pattern=r"^E\d+$")
    name: str
    type: Literal["person", "org", "product", "goal", "metric", "place", "time", "concept"]
    aliases: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class Relation(BaseModel):
    id: str = Field(..., pattern=r"^R\d+$")
    subj: str  # Entity ID
    pred: str
    obj: Union[str, Any]  # Entity ID or literal
    confidence: float = Field(..., ge=0.0, le=1.0)
    hypothesis: bool = True
    evidence: List[Evidence] = Field(default_factory=list)
    temporal: Optional[TemporalInfo] = None


class QuestionMapping(BaseModel):
    question_id: str
    mentions: List[str]  # Entity IDs
    claims: List[str]    # Relation IDs


class OntologyUpdate(BaseModel):
    entities: List[Entity]
    relations: List[Relation]
    mappings: List[QuestionMapping]


class GodmodeResponse(BaseModel):
    chat_reply: str = Field(..., max_length=800)  # Generous limit for max_tokens_reply
    graph_update: GraphUpdate
    ontology_update: OntologyUpdate


# Command schemas
class InitCommand(BaseModel):
    current_question: str
    context: Optional[str] = None
    budgets: Optional[Dict[str, Any]] = None


class AdvanceCommand(BaseModel):
    node_id: str
    user_answer: Optional[str] = None


class ContinueCommand(BaseModel):
    thread_id: str


class SummarizeCommand(BaseModel):
    thread_id: str


class RegraftCommand(BaseModel):
    from_node_id: str
    to_lane_id: str


class MergeCommand(BaseModel):
    thread_ids: List[str]


# Unified command interface
class Command(BaseModel):
    command_type: Literal["INIT", "ADVANCE", "CONTINUE", "SUMMARIZE", "REGRAFT", "MERGE"]
    data: Union[InitCommand, AdvanceCommand, ContinueCommand, SummarizeCommand, RegraftCommand, MergeCommand]