"""Schema validation utilities for GODMODE."""

from typing import Dict, Any, List, Optional
from jsonschema import validate, ValidationError, Draft7Validator
from ..models.responses import GodmodeResponse
from .schemas import GODMODE_RESPONSE_SCHEMA


class SchemaValidator:
    """Schema validator for GODMODE responses."""
    
    def __init__(self):
        self.validator = Draft7Validator(GODMODE_RESPONSE_SCHEMA)
    
    def validate_response(self, response: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a GODMODE response.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            self.validator.validate(response)
            
            # Additional custom validations
            errors.extend(self._validate_invariants(response))
            
            return len(errors) == 0, errors
            
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            return False, errors
    
    def _validate_invariants(self, response: Dict[str, Any]) -> List[str]:
        """Validate GODMODE-specific invariants."""
        errors = []
        
        graph_update = response.get("graph_update", {})
        
        # Validate DAG structure (no cycles)
        errors.extend(self._validate_dag_structure(graph_update))
        
        # Validate level progression
        errors.extend(self._validate_level_progression(graph_update))
        
        # Validate no near-duplicates at same level
        errors.extend(self._validate_no_duplicates(graph_update))
        
        # Validate scenario lane count
        scenarios = graph_update.get("scenarios", [])
        if not (3 <= len(scenarios) <= 5):
            errors.append(f"Must have 3-5 scenario lanes, got {len(scenarios)}")
        
        return errors
    
    def _validate_dag_structure(self, graph_update: Dict[str, Any]) -> List[str]:
        """Validate that the graph is a DAG (no cycles)."""
        errors = []
        
        # Collect all nodes and their dependencies
        all_nodes = {}
        
        # Add priors
        for node in graph_update.get("priors", []):
            all_nodes[node["id"]] = node["builds_on"]
        
        # Add scenario lanes
        for scenario in graph_update.get("scenarios", []):
            for node in scenario.get("lane", []):
                all_nodes[node["id"]] = node["builds_on"]
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
                
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for parent in all_nodes.get(node_id, []):
                if parent in all_nodes and has_cycle(parent):
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in all_nodes:
            if node_id not in visited and has_cycle(node_id):
                errors.append(f"Cycle detected involving node {node_id}")
                break
        
        return errors
    
    def _validate_level_progression(self, graph_update: Dict[str, Any]) -> List[str]:
        """Validate that levels increase along builds_on chains."""
        errors = []
        
        # Collect all nodes
        all_nodes = {}
        
        for node in graph_update.get("priors", []):
            all_nodes[node["id"]] = node
        
        for scenario in graph_update.get("scenarios", []):
            for node in scenario.get("lane", []):
                all_nodes[node["id"]] = node
        
        # Check level progression
        for node_id, node in all_nodes.items():
            for parent_id in node.get("builds_on", []):
                if parent_id in all_nodes:
                    parent = all_nodes[parent_id]
                    if node["level"] <= parent["level"]:
                        errors.append(
                            f"Node {node_id} level {node['level']} must be > parent {parent_id} level {parent['level']}"
                        )
        
        return errors
    
    def _validate_no_duplicates(self, graph_update: Dict[str, Any]) -> List[str]:
        """Validate no near-duplicate text at same level within same lane."""
        errors = []
        
        # Check priors
        priors_by_level = {}
        for node in graph_update.get("priors", []):
            level = node["level"]
            if level not in priors_by_level:
                priors_by_level[level] = []
            priors_by_level[level].append(node["text"])
        
        for level, texts in priors_by_level.items():
            if len(texts) != len(set(texts)):
                errors.append(f"Duplicate text found in priors at level {level}")
        
        # Check scenario lanes
        for scenario in graph_update.get("scenarios", []):
            lane_by_level = {}
            for node in scenario.get("lane", []):
                level = node["level"]
                if level not in lane_by_level:
                    lane_by_level[level] = []
                lane_by_level[level].append(node["text"])
            
            for level, texts in lane_by_level.items():
                if len(texts) != len(set(texts)):
                    errors.append(f"Duplicate text found in lane {scenario['id']} at level {level}")
        
        return errors