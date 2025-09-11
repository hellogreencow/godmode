"""Response models for GODMODE outputs."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .core import Question, Lane, Thread, Entity, Relation, Mapping


class Meta(BaseModel):
    """Metadata about the processing."""
    version: str = Field(default="1.0", description="Schema version")
    budgets_used: Dict[str, Any] = Field(..., description="Actual budgets consumed")
    notes: Optional[str] = Field(None, description="Optional processing notes")


class GraphUpdate(BaseModel):
    """Graph update containing priors, scenarios, and threads."""
    current_question: str = Field(..., description="The current question being analyzed")
    priors: List[Question] = Field(..., description="Backward reasoning ladder")
    scenarios: List[Lane] = Field(..., description="Forward reasoning scenario lanes")
    threads: List[Thread] = Field(..., description="Active conversation threads")
    meta: Meta = Field(..., description="Processing metadata")


class OntologyUpdate(BaseModel):
    """Ontology update containing entities, relations, and mappings."""
    entities: List[Entity] = Field(..., description="Knowledge graph entities")
    relations: List[Relation] = Field(..., description="Knowledge graph relations")
    mappings: List[Mapping] = Field(..., description="Question to ontology mappings")


class GodmodeResponse(BaseModel):
    """Complete GODMODE response."""
    chat_reply: str = Field(..., description="Concise tactical response")
    graph_update: GraphUpdate = Field(..., description="Graph structure update")
    ontology_update: OntologyUpdate = Field(..., description="Ontology knowledge update")
    
    class Config:
        json_encoders = {
            # Handle datetime serialization if needed
        }