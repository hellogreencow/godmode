"""Ontology manager for GODMODE knowledge graph."""

import asyncio
from typing import List, Dict, Set, Optional
from ...models.core import Question, Entity, Relation, Mapping, EntityType
from ...models.responses import OntologyUpdate
from .extractor import EntityExtractor, RelationExtractor
from .graph import KnowledgeGraph


class OntologyManager:
    """
    Manages the knowledge graph ontology for GODMODE.
    
    Extracts entities, relations, and mappings from questions and context.
    Maintains consistency and provides query capabilities.
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.knowledge_graph = KnowledgeGraph()
        
        # ID counters
        self._entity_counter = 0
        self._relation_counter = 0
    
    async def extract_from_question(
        self,
        current_question: str,
        context: Optional[str],
        questions: List[Question]
    ) -> OntologyUpdate:
        """
        Extract ontology elements from a question and related context.
        
        Returns complete ontology update with entities, relations, and mappings.
        """
        # Extract entities and relations in parallel
        entities_task = self.entity_extractor.extract_entities(
            current_question, context, questions
        )
        relations_task = self.relation_extractor.extract_relations(
            current_question, context, questions
        )
        
        extracted_entities, extracted_relations = await asyncio.gather(
            entities_task, relations_task
        )
        
        # Assign IDs and add to knowledge graph
        entities = []
        for entity_data in extracted_entities:
            entity = Entity(
                id=f"E{self._next_entity_id()}",
                name=entity_data["name"],
                type=entity_data["type"],
                aliases=entity_data.get("aliases", []),
                confidence=entity_data.get("confidence", 0.8)
            )
            entities.append(entity)
            self.knowledge_graph.add_entity(entity)
        
        relations = []
        for relation_data in extracted_relations:
            relation = Relation(
                id=f"R{self._next_relation_id()}",
                subj=relation_data["subj"],
                pred=relation_data["pred"],
                obj=relation_data["obj"],
                confidence=relation_data.get("confidence", 0.7),
                hypothesis=relation_data.get("hypothesis", True),
                evidence=relation_data.get("evidence", []),
                temporal=relation_data.get("temporal")
            )
            relations.append(relation)
            self.knowledge_graph.add_relation(relation)
        
        # Create mappings between questions and ontology elements
        mappings = self._create_mappings(questions, entities, relations)
        
        return OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=mappings
        )
    
    async def update_from_expansion(self, questions: List[Question]) -> OntologyUpdate:
        """Update ontology from expanded questions."""
        # Extract new entities and relations from expanded questions
        all_text = " ".join(q.text for q in questions)
        
        extracted_entities = await self.entity_extractor.extract_entities_from_text(all_text)
        extracted_relations = await self.relation_extractor.extract_relations_from_text(all_text)
        
        # Convert to proper objects
        entities = []
        for entity_data in extracted_entities:
            entity = Entity(
                id=f"E{self._next_entity_id()}",
                name=entity_data["name"],
                type=entity_data["type"],
                aliases=entity_data.get("aliases", []),
                confidence=entity_data.get("confidence", 0.7)
            )
            entities.append(entity)
            self.knowledge_graph.add_entity(entity)
        
        relations = []
        for relation_data in extracted_relations:
            relation = Relation(
                id=f"R{self._next_relation_id()}",
                subj=relation_data["subj"],
                pred=relation_data["pred"],
                obj=relation_data["obj"],
                confidence=relation_data.get("confidence", 0.6),
                hypothesis=relation_data.get("hypothesis", True),
                evidence=relation_data.get("evidence", [])
            )
            relations.append(relation)
            self.knowledge_graph.add_relation(relation)
        
        # Create mappings
        mappings = self._create_mappings(questions, entities, relations)
        
        return OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=mappings
        )
    
    async def get_current_state(self) -> OntologyUpdate:
        """Get current state of the ontology."""
        return OntologyUpdate(
            entities=list(self.knowledge_graph.entities.values()),
            relations=list(self.knowledge_graph.relations.values()),
            mappings=list(self.knowledge_graph.mappings.values())
        )
    
    def _create_mappings(
        self, 
        questions: List[Question],
        entities: List[Entity],
        relations: List[Relation]
    ) -> List[Mapping]:
        """Create mappings between questions and ontology elements."""
        mappings = []
        
        for question in questions:
            question_text = question.text.lower()
            
            # Find mentioned entities
            mentioned_entities = []
            for entity in entities:
                if (entity.name.lower() in question_text or 
                    any(alias.lower() in question_text for alias in entity.aliases)):
                    mentioned_entities.append(entity.id)
            
            # Find claimed relations
            claimed_relations = []
            for relation in relations:
                # Simple heuristic: if both subject and object entities are mentioned
                subj_entity = next((e for e in entities if e.id == relation.subj), None)
                obj_entity = next((e for e in entities if e.id == relation.obj), None)
                
                if (subj_entity and obj_entity and
                    subj_entity.name.lower() in question_text and
                    str(relation.obj).lower() in question_text):
                    claimed_relations.append(relation.id)
            
            if mentioned_entities or claimed_relations:
                mapping = Mapping(
                    question_id=question.id,
                    mentions=mentioned_entities,
                    claims=claimed_relations
                )
                mappings.append(mapping)
                self.knowledge_graph.add_mapping(mapping)
        
        return mappings
    
    def query_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Query entities by type."""
        return [e for e in self.knowledge_graph.entities.values() if e.type == entity_type]
    
    def query_relations_by_predicate(self, predicate: str) -> List[Relation]:
        """Query relations by predicate."""
        return [r for r in self.knowledge_graph.relations.values() if r.pred == predicate]
    
    def find_related_entities(self, entity_id: str) -> List[Entity]:
        """Find entities related to a given entity."""
        related_ids = set()
        
        # Find entities connected via relations
        for relation in self.knowledge_graph.relations.values():
            if relation.subj == entity_id:
                if isinstance(relation.obj, str) and relation.obj.startswith('E'):
                    related_ids.add(relation.obj)
            elif relation.obj == entity_id:
                related_ids.add(relation.subj)
        
        return [self.knowledge_graph.entities[eid] for eid in related_ids 
                if eid in self.knowledge_graph.entities]
    
    def get_entity_mentions(self, entity_id: str) -> List[str]:
        """Get questions that mention a specific entity."""
        question_ids = []
        for mapping in self.knowledge_graph.mappings.values():
            if entity_id in mapping.mentions:
                question_ids.append(mapping.question_id)
        return question_ids
    
    def _next_entity_id(self) -> int:
        """Get next entity ID."""
        self._entity_counter += 1
        return self._entity_counter
    
    def _next_relation_id(self) -> int:
        """Get next relation ID."""
        self._relation_counter += 1
        return self._relation_counter