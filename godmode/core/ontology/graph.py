"""Knowledge graph implementation for GODMODE ontology."""

from typing import Dict, List, Set, Optional, Tuple
import networkx as nx
from ...models.core import Entity, Relation, Mapping


class KnowledgeGraph:
    """
    In-memory knowledge graph for GODMODE ontology.
    
    Stores entities, relations, and mappings with query capabilities.
    Uses NetworkX for graph operations and analysis.
    """
    
    def __init__(self):
        # Core storage
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.mappings: Dict[str, Mapping] = {}
        
        # NetworkX graph for analysis
        self.graph = nx.MultiDiGraph()
        
        # Indexes for fast queries
        self.entity_by_name: Dict[str, str] = {}  # name -> entity_id
        self.entity_by_type: Dict[str, Set[str]] = {}  # type -> {entity_ids}
        self.relations_by_predicate: Dict[str, Set[str]] = {}  # predicate -> {relation_ids}
        self.entity_relations: Dict[str, Set[str]] = {}  # entity_id -> {relation_ids}
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph."""
        # Store entity
        self.entities[entity.id] = entity
        
        # Update indexes
        self.entity_by_name[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            self.entity_by_name[alias.lower()] = entity.id
        
        if entity.type.value not in self.entity_by_type:
            self.entity_by_type[entity.type.value] = set()
        self.entity_by_type[entity.type.value].add(entity.id)
        
        # Add to NetworkX graph
        self.graph.add_node(entity.id, **{
            "name": entity.name,
            "type": entity.type.value,
            "confidence": entity.confidence
        })
    
    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the knowledge graph."""
        # Store relation
        self.relations[relation.id] = relation
        
        # Update indexes
        if relation.pred not in self.relations_by_predicate:
            self.relations_by_predicate[relation.pred] = set()
        self.relations_by_predicate[relation.pred].add(relation.id)
        
        # Update entity relations index
        if relation.subj not in self.entity_relations:
            self.entity_relations[relation.subj] = set()
        self.entity_relations[relation.subj].add(relation.id)
        
        # Add to NetworkX graph if both subj and obj are entities
        if (relation.subj in self.entities and 
            isinstance(relation.obj, str) and relation.obj in self.entities):
            
            self.graph.add_edge(
                relation.subj, 
                relation.obj,
                key=relation.id,
                predicate=relation.pred,
                confidence=relation.confidence,
                hypothesis=relation.hypothesis
            )
    
    def add_mapping(self, mapping: Mapping) -> None:
        """Add a question-ontology mapping."""
        self.mappings[mapping.question_id] = mapping
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name (case-insensitive)."""
        entity_id = self.entity_by_name.get(name.lower())
        return self.entities.get(entity_id) if entity_id else None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids]
    
    def get_relations_by_predicate(self, predicate: str) -> List[Relation]:
        """Get all relations with a specific predicate."""
        relation_ids = self.relations_by_predicate.get(predicate, set())
        return [self.relations[rid] for rid in relation_ids]
    
    def get_entity_relations(self, entity_id: str) -> List[Relation]:
        """Get all relations where entity is the subject."""
        relation_ids = self.entity_relations.get(entity_id, set())
        relations = [self.relations[rid] for rid in relation_ids]
        
        # Also find relations where entity is the object
        for relation in self.relations.values():
            if relation.obj == entity_id:
                relations.append(relation)
        
        return relations
    
    def find_connected_entities(self, entity_id: str, max_hops: int = 2) -> List[Entity]:
        """Find entities connected to the given entity within max_hops."""
        if entity_id not in self.graph:
            return []
        
        connected_ids = set()
        
        # Use NetworkX to find connected nodes
        try:
            # Get nodes within max_hops distance
            for target_id in self.graph.nodes():
                if target_id != entity_id:
                    try:
                        path_length = nx.shortest_path_length(
                            self.graph, entity_id, target_id
                        )
                        if path_length <= max_hops:
                            connected_ids.add(target_id)
                    except nx.NetworkXNoPath:
                        continue
        except Exception:
            # Fallback to direct neighbors
            connected_ids = set(self.graph.neighbors(entity_id))
        
        return [self.entities[eid] for eid in connected_ids if eid in self.entities]
    
    def find_shortest_path(self, entity1_id: str, entity2_id: str) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        try:
            return nx.shortest_path(self.graph, entity1_id, entity2_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_entity_centrality(self, entity_id: str) -> float:
        """Get centrality score for an entity (how connected it is)."""
        if entity_id not in self.graph or self.graph.number_of_nodes() < 2:
            return 0.0
        
        try:
            centrality_scores = nx.degree_centrality(self.graph)
            return centrality_scores.get(entity_id, 0.0)
        except Exception:
            return 0.0
    
    def find_similar_entities(self, entity_id: str, similarity_threshold: float = 0.5) -> List[Entity]:
        """Find entities similar to the given entity."""
        if entity_id not in self.entities:
            return []
        
        target_entity = self.entities[entity_id]
        similar_entities = []
        
        for other_id, other_entity in self.entities.items():
            if other_id == entity_id:
                continue
            
            similarity = self._calculate_entity_similarity(target_entity, other_entity)
            if similarity >= similarity_threshold:
                similar_entities.append(other_entity)
        
        # Sort by similarity (approximate)
        similar_entities.sort(key=lambda e: self._calculate_entity_similarity(target_entity, e), reverse=True)
        
        return similar_entities
    
    def get_question_entities(self, question_id: str) -> List[Entity]:
        """Get entities mentioned in a specific question."""
        mapping = self.mappings.get(question_id)
        if not mapping:
            return []
        
        return [self.entities[eid] for eid in mapping.mentions if eid in self.entities]
    
    def get_question_relations(self, question_id: str) -> List[Relation]:
        """Get relations claimed in a specific question."""
        mapping = self.mappings.get(question_id)
        if not mapping:
            return []
        
        return [self.relations[rid] for rid in mapping.claims if rid in self.relations]
    
    def get_graph_statistics(self) -> Dict[str, int]:
        """Get basic statistics about the knowledge graph."""
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_mappings": len(self.mappings),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "entity_types": len(self.entity_by_type),
            "relation_predicates": len(self.relations_by_predicate)
        }
    
    def export_to_triples(self) -> List[Tuple[str, str, str]]:
        """Export knowledge graph as RDF-like triples."""
        triples = []
        
        # Entity type triples
        for entity in self.entities.values():
            triples.append((entity.id, "rdf:type", entity.type.value))
            triples.append((entity.id, "rdfs:label", entity.name))
            
            for alias in entity.aliases:
                triples.append((entity.id, "skos:altLabel", alias))
        
        # Relation triples
        for relation in self.relations.values():
            triples.append((relation.subj, relation.pred, str(relation.obj)))
            
            # Add metadata triples
            triples.append((relation.id, "confidence", str(relation.confidence)))
            triples.append((relation.id, "hypothesis", str(relation.hypothesis)))
        
        return triples
    
    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between two entities."""
        similarity = 0.0
        
        # Type similarity
        if entity1.type == entity2.type:
            similarity += 0.4
        
        # Name similarity (simple word overlap)
        words1 = set(entity1.name.lower().split())
        words2 = set(entity2.name.lower().split())
        
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            similarity += 0.3 * word_overlap
        
        # Alias similarity
        all_names1 = {entity1.name.lower()} | {alias.lower() for alias in entity1.aliases}
        all_names2 = {entity2.name.lower()} | {alias.lower() for alias in entity2.aliases}
        
        name_overlap = len(all_names1 & all_names2)
        if name_overlap > 0:
            similarity += 0.3
        
        return similarity