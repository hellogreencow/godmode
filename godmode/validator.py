"""
Invariant validation for GODMODE responses
"""

from typing import List, Dict, Set, Optional
import networkx as nx

from .schemas import GodmodeResponse, QuestionNode, ScenarioLane, Thread


class ValidationError(Exception):
    """Raised when invariant validation fails"""
    pass


class InvariantValidator:
    """Validates GODMODE responses against system invariants"""
    
    def validate_response(self, response: GodmodeResponse) -> None:
        """Validate complete response against all invariants"""
        try:
            self.validate_graph_structure(response.graph_update)
            self.validate_progressive_levels(response.graph_update)
            self.validate_cognitive_progression(response.graph_update)
            self.validate_no_duplicates(response.graph_update)
            self.validate_builds_on_relationships(response.graph_update)
            self.validate_delta_nuances(response.graph_update)
        except ValidationError as e:
            # In production, could attempt repair or graceful degradation
            raise e
    
    def validate_graph_structure(self, graph_update) -> None:
        """Ensure graph forms a valid DAG (no cycles)"""
        # Build graph from all nodes
        G = nx.DiGraph()
        all_nodes = []
        
        # Add prior nodes
        all_nodes.extend(graph_update.priors)
        
        # Add scenario nodes
        for scenario in graph_update.scenarios:
            all_nodes.extend(scenario.lane)
        
        # Add nodes to graph
        for node in all_nodes:
            G.add_node(node.id)
            for parent_id in node.builds_on:
                G.add_edge(parent_id, node.id)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            raise ValidationError(f"Graph contains cycles: {cycles}")
    
    def validate_progressive_levels(self, graph_update) -> None:
        """Ensure levels increase along builds_on chains"""
        all_nodes = {}
        
        # Index all nodes
        for node in graph_update.priors:
            all_nodes[node.id] = node
        
        for scenario in graph_update.scenarios:
            for node in scenario.lane:
                all_nodes[node.id] = node
        
        # Validate level progression
        for node in all_nodes.values():
            for parent_id in node.builds_on:
                parent = all_nodes.get(parent_id)
                if parent and parent.level >= node.level:
                    raise ValidationError(
                        f"Node {node.id} (level {node.level}) must have higher level than parent {parent_id} (level {parent.level})"
                    )
    
    def validate_cognitive_progression(self, graph_update) -> None:
        """Validate cognitive move progression within ladders"""
        cognitive_order = ["define", "scope", "quantify", "compare", "simulate", "decide", "commit"]
        order_map = {move: i for i, move in enumerate(cognitive_order)}
        
        # Validate priors
        self._validate_move_sequence(graph_update.priors, order_map, "priors")
        
        # Validate each scenario lane
        for scenario in graph_update.scenarios:
            self._validate_move_sequence(scenario.lane, order_map, f"scenario {scenario.id}")
    
    def _validate_move_sequence(self, nodes: List[QuestionNode], order_map: Dict[str, int], context: str) -> None:
        """Validate cognitive move sequence for a list of nodes"""
        if not nodes:
            return
        
        # Check that moves generally progress (allowing some flexibility)
        for i in range(1, len(nodes)):
            current_order = order_map.get(nodes[i].cognitive_move, -1)
            prev_order = order_map.get(nodes[i-1].cognitive_move, -1)
            
            # Allow same level or progression, but not major regression
            if current_order != -1 and prev_order != -1 and current_order < prev_order - 1:
                raise ValidationError(
                    f"Invalid cognitive regression in {context}: {nodes[i-1].cognitive_move} → {nodes[i].cognitive_move}"
                )
    
    def validate_no_duplicates(self, graph_update) -> None:
        """Ensure no near-duplicate questions at the same level within lanes"""
        # Check priors
        self._check_duplicates_in_sequence(graph_update.priors, "priors")
        
        # Check each scenario lane
        for scenario in graph_update.scenarios:
            self._check_duplicates_in_sequence(scenario.lane, f"scenario {scenario.id}")
    
    def _check_duplicates_in_sequence(self, nodes: List[QuestionNode], context: str) -> None:
        """Check for duplicates within a sequence of nodes"""
        level_questions = {}
        
        for node in nodes:
            level = node.level
            if level not in level_questions:
                level_questions[level] = []
            
            # Simple duplicate detection based on normalized text
            normalized = " ".join(node.text.lower().split())
            
            for existing_text in level_questions[level]:
                if self._texts_similar(normalized, existing_text):
                    raise ValidationError(
                        f"Duplicate questions at level {level} in {context}: '{node.text}'"
                    )
            
            level_questions[level].append(normalized)
    
    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar (simple word overlap)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union >= threshold
    
    def validate_builds_on_relationships(self, graph_update) -> None:
        """Validate builds_on relationships are valid"""
        all_node_ids = set()
        
        # Collect all valid node IDs
        for node in graph_update.priors:
            all_node_ids.add(node.id)
        
        for scenario in graph_update.scenarios:
            for node in scenario.lane:
                all_node_ids.add(node.id)
        
        # Validate builds_on references
        all_nodes = graph_update.priors[:]
        for scenario in graph_update.scenarios:
            all_nodes.extend(scenario.lane)
        
        for node in all_nodes:
            # Nodes at level > 1 must build on something
            if node.level > 1 and not node.builds_on:
                raise ValidationError(f"Node {node.id} at level {node.level} must build on at least one earlier node")
            
            # All builds_on references must be valid
            for parent_id in node.builds_on:
                if parent_id not in all_node_ids:
                    raise ValidationError(f"Node {node.id} builds_on invalid node {parent_id}")
    
    def validate_delta_nuances(self, graph_update) -> None:
        """Validate that each node has a meaningful delta nuance"""
        all_nodes = graph_update.priors[:]
        for scenario in graph_update.scenarios:
            all_nodes.extend(scenario.lane)
        
        for node in all_nodes:
            if not node.delta_nuance or len(node.delta_nuance.strip()) < 10:
                raise ValidationError(f"Node {node.id} has insufficient delta_nuance: '{node.delta_nuance}'")
            
            # Check that delta_nuance describes what's new
            if not any(keyword in node.delta_nuance.lower() for keyword in 
                      ["adds", "introduces", "clarifies", "establishes", "explores", "focuses", "addresses"]):
                raise ValidationError(f"Node {node.id} delta_nuance doesn't describe what's new: '{node.delta_nuance}'")
    
    def repair_response(self, response: GodmodeResponse) -> GodmodeResponse:
        """Attempt to repair common validation issues"""
        # This would implement repair strategies for common issues
        # For now, just return the original response
        return response