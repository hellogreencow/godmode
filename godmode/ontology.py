"""
Ontology and Knowledge Graph management for GODMODE
"""

import re
import uuid
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .schemas import Entity, Relation, QuestionMapping, Evidence, TemporalInfo


class OntologyExtractor:
    """Extracts entities, relations, and mappings from questions and context"""
    
    def __init__(self):
        # Common entity patterns
        self.entity_patterns = {
            "person": [
                r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",  # John Smith
                r"\b(?:CEO|CTO|VP|Director|Manager|President)\s[A-Z][a-z]+\b",  # CEO Johnson
            ],
            "org": [
                r"\b[A-Z][a-zA-Z]*\s(?:Inc|LLC|Corp|Company|Corporation|Ltd)\b",  # Apple Inc
                r"\b(?:Google|Apple|Microsoft|Amazon|Meta|Tesla|Netflix)\b",  # Known companies
            ],
            "product": [
                r"\b[A-Z][a-zA-Z]*\s(?:app|software|platform|tool|service|product)\b",
            ],
            "metric": [
                r"\b(?:revenue|profit|growth|ROI|conversion|retention|churn|CAC|LTV)\b",
                r"\b\d+%\b",  # Percentages
                r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b",  # Money amounts
            ],
            "time": [
                r"\b(?:Q[1-4]|quarter|monthly|yearly|annually)\b",
                r"\b(?:2024|2025|2026)\b",  # Years
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            ],
            "place": [
                r"\b[A-Z][a-z]+,\s[A-Z]{2}\b",  # City, ST
                r"\b(?:US|USA|Europe|Asia|California|New York|Texas)\b",
            ],
        }
        
        # Relation patterns
        self.relation_patterns = [
            (r"(\w+)\s+(?:owns|acquired|bought)\s+(\w+)", "owns"),
            (r"(\w+)\s+(?:works at|employed by)\s+(\w+)", "works_at"),
            (r"(\w+)\s+(?:competes with|rival of)\s+(\w+)", "competes_with"),
            (r"(\w+)\s+(?:costs|priced at)\s+(\$\d+)", "costs"),
            (r"(\w+)\s+(?:located in|based in)\s+(\w+)", "located_in"),
            (r"(\w+)\s+(?:reports to|managed by)\s+(\w+)", "reports_to"),
        ]
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        entities = []
        entity_counter = defaultdict(int)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 2:  # Filter very short matches
                        entity_counter[entity_type] += 1
                        entity_id = f"E{sum(entity_counter.values())}"
                        
                        entities.append(Entity(
                            id=entity_id,
                            name=entity_text,
                            type=entity_type,
                            aliases=[],
                            confidence=0.7  # Default confidence for pattern matches
                        ))
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities"""
        relations = []
        relation_counter = 0
        
        # Create entity lookup
        entity_lookup = {entity.name.lower(): entity.id for entity in entities}
        
        for pattern, relation_type in self.relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subj_text = match.group(1).strip()
                obj_text = match.group(2).strip()
                
                subj_id = entity_lookup.get(subj_text.lower())
                obj_id = entity_lookup.get(obj_text.lower())
                
                if subj_id and obj_id:
                    relation_counter += 1
                    relations.append(Relation(
                        id=f"R{relation_counter}",
                        subj=subj_id,
                        pred=relation_type,
                        obj=obj_id,
                        confidence=0.6,
                        hypothesis=True,  # Mark as hypothesis since extracted from patterns
                        evidence=[Evidence(
                            source_type="user",
                            snippet=match.group()
                        )]
                    ))
        
        return relations
    
    def create_question_mapping(self, question_id: str, question_text: str, 
                              entities: List[Entity], relations: List[Relation]) -> QuestionMapping:
        """Create mapping between question and ontology elements"""
        mentioned_entities = []
        claimed_relations = []
        
        question_lower = question_text.lower()
        
        # Find entities mentioned in the question
        for entity in entities:
            if entity.name.lower() in question_lower:
                mentioned_entities.append(entity.id)
        
        # Find relations that might be relevant to the question
        for relation in relations:
            # Check if relation subjects/objects are mentioned in question
            subj_entity = next((e for e in entities if e.id == relation.subj), None)
            obj_entity = next((e for e in entities if e.id == relation.obj), None)
            
            if (subj_entity and subj_entity.name.lower() in question_lower) or \
               (obj_entity and obj_entity.name.lower() in question_lower):
                claimed_relations.append(relation.id)
        
        return QuestionMapping(
            question_id=question_id,
            mentions=mentioned_entities,
            claims=claimed_relations
        )


class OntologyManager:
    """Manages the knowledge graph and ontology state"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.mappings: Dict[str, QuestionMapping] = {}  # question_id -> mapping
        self.extractor = OntologyExtractor()
    
    def update_from_question(self, question_id: str, question_text: str, 
                           context: str = "") -> Tuple[List[Entity], List[Relation], QuestionMapping]:
        """Update ontology from a new question and context"""
        full_text = f"{question_text} {context}".strip()
        
        # Extract new entities and relations
        new_entities = self.extractor.extract_entities(full_text)
        new_relations = self.extractor.extract_relations(full_text, new_entities)
        
        # Merge with existing ontology
        self._merge_entities(new_entities)
        self._merge_relations(new_relations)
        
        # Create question mapping
        all_entities = list(self.entities.values())
        all_relations = list(self.relations.values())
        mapping = self.extractor.create_question_mapping(
            question_id, question_text, all_entities, all_relations
        )
        self.mappings[question_id] = mapping
        
        return new_entities, new_relations, mapping
    
    def _merge_entities(self, new_entities: List[Entity]) -> None:
        """Merge new entities with existing ones, handling duplicates"""
        for entity in new_entities:
            # Check for existing entity with same name
            existing = None
            for existing_entity in self.entities.values():
                if existing_entity.name.lower() == entity.name.lower():
                    existing = existing_entity
                    break
            
            if existing:
                # Update confidence if new entity has higher confidence
                if entity.confidence > existing.confidence:
                    existing.confidence = entity.confidence
                # Merge aliases
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
            else:
                # Add new entity
                self.entities[entity.id] = entity
    
    def _merge_relations(self, new_relations: List[Relation]) -> None:
        """Merge new relations with existing ones"""
        for relation in new_relations:
            # Check for duplicate relations
            duplicate_found = False
            for existing_relation in self.relations.values():
                if (existing_relation.subj == relation.subj and 
                    existing_relation.pred == relation.pred and 
                    existing_relation.obj == relation.obj):
                    # Update confidence and evidence
                    if relation.confidence > existing_relation.confidence:
                        existing_relation.confidence = relation.confidence
                    existing_relation.evidence.extend(relation.evidence)
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                self.relations[relation.id] = relation
    
    def get_related_entities(self, entity_id: str, max_hops: int = 2) -> Set[str]:
        """Get entities related to the given entity within max_hops"""
        related = set()
        current_level = {entity_id}
        
        for hop in range(max_hops):
            next_level = set()
            for current_entity in current_level:
                # Find relations involving this entity
                for relation in self.relations.values():
                    if relation.subj == current_entity:
                        next_level.add(relation.obj)
                    elif relation.obj == current_entity:
                        next_level.add(relation.subj)
            
            related.update(next_level)
            current_level = next_level - related  # Only explore new entities
            
            if not current_level:  # No new entities to explore
                break
        
        return related
    
    def get_entity_context(self, entity_id: str) -> Dict[str, any]:
        """Get rich context for an entity including relations and properties"""
        entity = self.entities.get(entity_id)
        if not entity:
            return {}
        
        # Find all relations involving this entity
        incoming_relations = []
        outgoing_relations = []
        
        for relation in self.relations.values():
            if relation.subj == entity_id:
                outgoing_relations.append(relation)
            elif relation.obj == entity_id:
                incoming_relations.append(relation)
        
        # Get related entities
        related_entity_ids = self.get_related_entities(entity_id, max_hops=1)
        related_entities = [self.entities[eid] for eid in related_entity_ids if eid in self.entities]
        
        return {
            "entity": entity,
            "outgoing_relations": outgoing_relations,
            "incoming_relations": incoming_relations,
            "related_entities": related_entities,
            "total_connections": len(incoming_relations) + len(outgoing_relations)
        }
    
    def export_for_update(self) -> Tuple[List[Entity], List[Relation], List[QuestionMapping]]:
        """Export current state for ontology update"""
        return (
            list(self.entities.values()),
            list(self.relations.values()),
            list(self.mappings.values())
        )