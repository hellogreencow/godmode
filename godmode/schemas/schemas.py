"""JSON schemas for GODMODE data validation."""

import json
from typing import Dict, Any
from jsonschema import validate, ValidationError
from ..models.responses import GodmodeResponse


# Core response schema matching the SYSTEM specification
GODMODE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "chat_reply": {
            "type": "string",
            "description": "Concise tactical response (2-5 sentences)"
        },
        "graph_update": {
            "type": "object",
            "properties": {
                "current_question": {"type": "string"},
                "priors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "pattern": "^Q[P]?[0-9]+$"},
                            "text": {"type": "string"},
                            "level": {"type": "integer", "minimum": 1},
                            "cognitive_move": {
                                "type": "string",
                                "enum": ["define", "scope", "quantify", "compare", "simulate", "decide", "commit"]
                            },
                            "builds_on": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "delta_nuance": {"type": "string"},
                            "expected_info_gain": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["id", "text", "level", "cognitive_move", "builds_on", "delta_nuance", "expected_info_gain", "confidence", "tags"]
                    }
                },
                "scenarios": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "pattern": "^S-[A-Z]$"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "lane": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "pattern": "^Q[A-Z]?[0-9]+$"},
                                        "text": {"type": "string"},
                                        "level": {"type": "integer", "minimum": 1},
                                        "cognitive_move": {
                                            "type": "string",
                                            "enum": ["define", "scope", "quantify", "compare", "simulate", "decide", "commit"]
                                        },
                                        "builds_on": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "delta_nuance": {"type": "string"},
                                        "expected_info_gain": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "triggers": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string",
                                                        "enum": ["event", "metric", "time", "answer_change"]
                                                    },
                                                    "detail": {"type": "string"}
                                                },
                                                "required": ["type", "detail"]
                                            }
                                        },
                                        "natural_end": {"type": "boolean"},
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["id", "text", "level", "cognitive_move", "builds_on", "delta_nuance", "expected_info_gain", "confidence", "natural_end", "tags"]
                                }
                            },
                            "cross_links": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "from": {"type": "string"},
                                        "to": {"type": "string"},
                                        "type": {"type": "string", "enum": ["junction"]}
                                    },
                                    "required": ["from", "to", "type"]
                                }
                            }
                        },
                        "required": ["id", "name", "description", "lane", "cross_links"]
                    }
                },
                "threads": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "thread_id": {"type": "string", "pattern": "^T[0-9]+$"},
                            "origin_node_id": {"type": "string"},
                            "path": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "status": {
                                "type": "string",
                                "enum": ["active", "paused", "ended"]
                            },
                            "summary": {"type": "string", "maxLength": 280}
                        },
                        "required": ["thread_id", "origin_node_id", "path", "status", "summary"]
                    }
                },
                "meta": {
                    "type": "object",
                    "properties": {
                        "version": {"type": "string"},
                        "budgets_used": {"type": "object"},
                        "notes": {"type": "string"}
                    },
                    "required": ["version", "budgets_used"]
                }
            },
            "required": ["current_question", "priors", "scenarios", "threads", "meta"]
        },
        "ontology_update": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "pattern": "^E[0-9]+$"},
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["person", "org", "product", "goal", "metric", "place", "time", "concept"]
                            },
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["id", "name", "type", "aliases", "confidence"]
                    }
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "pattern": "^R[0-9]+$"},
                            "subj": {"type": "string"},
                            "pred": {"type": "string"},
                            "obj": {"type": ["string", "number", "boolean"]},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "hypothesis": {"type": "boolean"},
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_type": {
                                            "type": "string",
                                            "enum": ["user", "memory", "citation"]
                                        },
                                        "source_id": {"type": "string"},
                                        "snippet": {"type": "string"},
                                        "note": {"type": "string"}
                                    },
                                    "required": ["source_type"]
                                }
                            },
                            "temporal": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string", "format": "date-time"},
                                    "end": {"type": "string", "format": "date-time"},
                                    "grain": {
                                        "type": "string",
                                        "enum": ["day", "week", "month", "quarter", "year"]
                                    }
                                }
                            }
                        },
                        "required": ["id", "subj", "pred", "obj", "confidence", "hypothesis", "evidence"]
                    }
                },
                "mappings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string"},
                            "mentions": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "claims": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["question_id", "mentions", "claims"]
                    }
                }
            },
            "required": ["entities", "relations", "mappings"]
        }
    },
    "required": ["chat_reply", "graph_update", "ontology_update"]
}


def get_schema(schema_name: str) -> Dict[str, Any]:
    """Get a specific schema by name."""
    schemas = {
        "godmode_response": GODMODE_RESPONSE_SCHEMA,
    }
    return schemas.get(schema_name, {})


def validate_response(data: Dict[str, Any]) -> bool:
    """Validate a GODMODE response against the schema."""
    try:
        validate(instance=data, schema=GODMODE_RESPONSE_SCHEMA)
        return True
    except ValidationError as e:
        print(f"Schema validation error: {e}")
        return False