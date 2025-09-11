"""Cognitive move progression validation and utilities."""

from typing import Dict, Set, List
from ...models.core import CognitiveMove


class CognitiveMoveProgression:
    """
    Manages cognitive move progression: define → scope → quantify → compare → simulate → decide → commit
    
    Ensures questions follow logical progression and validates transitions.
    """
    
    def __init__(self):
        # Define valid transitions between cognitive moves
        self.valid_transitions: Dict[CognitiveMove, Set[CognitiveMove]] = {
            CognitiveMove.DEFINE: {
                CognitiveMove.SCOPE,
                CognitiveMove.QUANTIFY,  # Can skip scope if definition is clear
            },
            CognitiveMove.SCOPE: {
                CognitiveMove.QUANTIFY,
                CognitiveMove.COMPARE,  # Can skip quantify for qualitative comparisons
            },
            CognitiveMove.QUANTIFY: {
                CognitiveMove.COMPARE,
                CognitiveMove.SIMULATE,  # Can skip compare if metrics are clear
            },
            CognitiveMove.COMPARE: {
                CognitiveMove.SIMULATE,
                CognitiveMove.DECIDE,  # Can skip simulate if comparison is decisive
            },
            CognitiveMove.SIMULATE: {
                CognitiveMove.DECIDE,
                CognitiveMove.COMMIT,  # Can skip decide if simulation is conclusive
            },
            CognitiveMove.DECIDE: {
                CognitiveMove.COMMIT,
            },
            CognitiveMove.COMMIT: set(),  # Terminal state
        }
        
        # Move ordering for level assignment
        self.move_order = [
            CognitiveMove.DEFINE,
            CognitiveMove.SCOPE,
            CognitiveMove.QUANTIFY,
            CognitiveMove.COMPARE,
            CognitiveMove.SIMULATE,
            CognitiveMove.DECIDE,
            CognitiveMove.COMMIT,
        ]
    
    def is_valid_progression(self, from_move: CognitiveMove, to_move: CognitiveMove) -> bool:
        """Check if transition from one cognitive move to another is valid."""
        return to_move in self.valid_transitions.get(from_move, set())
    
    def get_next_moves(self, current_move: CognitiveMove) -> Set[CognitiveMove]:
        """Get valid next moves from current cognitive move."""
        return self.valid_transitions.get(current_move, set())
    
    def get_move_level(self, move: CognitiveMove) -> int:
        """Get the typical level for a cognitive move (1-based)."""
        try:
            return self.move_order.index(move) + 1
        except ValueError:
            return 1  # Default to level 1 if move not found
    
    def is_terminal_move(self, move: CognitiveMove) -> bool:
        """Check if this is a terminal cognitive move."""
        return move == CognitiveMove.COMMIT or not self.valid_transitions.get(move)
    
    def suggest_next_move(self, current_move: CognitiveMove, context: str = "") -> CognitiveMove:
        """Suggest the most appropriate next cognitive move."""
        valid_next = self.get_next_moves(current_move)
        
        if not valid_next:
            return current_move  # No valid transitions
        
        # Default to the first valid move in progression order
        for move in self.move_order:
            if move in valid_next:
                return move
        
        # Fallback to any valid move
        return next(iter(valid_next))
    
    def get_progression_path(self, start_move: CognitiveMove, end_move: CognitiveMove) -> List[CognitiveMove]:
        """Get a valid progression path between two moves."""
        if start_move == end_move:
            return [start_move]
        
        # Simple BFS to find shortest path
        queue = [(start_move, [start_move])]
        visited = {start_move}
        
        while queue:
            current_move, path = queue.pop(0)
            
            for next_move in self.get_next_moves(current_move):
                if next_move == end_move:
                    return path + [next_move]
                
                if next_move not in visited:
                    visited.add(next_move)
                    queue.append((next_move, path + [next_move]))
        
        # No valid path found
        return []
    
    def validate_ladder_progression(self, questions: List) -> List[str]:
        """
        Validate that a ladder follows proper cognitive move progression.
        
        Returns list of validation errors.
        """
        errors = []
        
        if not questions:
            return errors
        
        # Check each question's move progression
        for i, question in enumerate(questions):
            # Check if question builds on valid parents
            for parent_id in question.builds_on:
                parent = next((q for q in questions if q.id == parent_id), None)
                if parent:
                    if not self.is_valid_progression(parent.cognitive_move, question.cognitive_move):
                        errors.append(
                            f"Invalid progression from {parent.cognitive_move} to {question.cognitive_move} "
                            f"(parent {parent_id} → {question.id})"
                        )
        
        return errors
    
    def get_move_description(self, move: CognitiveMove) -> str:
        """Get human-readable description of a cognitive move."""
        descriptions = {
            CognitiveMove.DEFINE: "Establish clear definitions and terminology",
            CognitiveMove.SCOPE: "Set boundaries, constraints, and context",
            CognitiveMove.QUANTIFY: "Add metrics, measurements, and thresholds",
            CognitiveMove.COMPARE: "Compare alternatives and trade-offs",
            CognitiveMove.SIMULATE: "Model outcomes and test scenarios",
            CognitiveMove.DECIDE: "Make decisions based on analysis",
            CognitiveMove.COMMIT: "Take action and commit to decisions",
        }
        return descriptions.get(move, "Unknown cognitive move")
    
    def get_move_delta_nuance_template(self, move: CognitiveMove) -> str:
        """Get template for delta_nuance based on cognitive move."""
        templates = {
            CognitiveMove.DEFINE: "Establishes definition of {concept}",
            CognitiveMove.SCOPE: "Adds scope constraint: {constraint}",
            CognitiveMove.QUANTIFY: "Introduces metric: {metric}",
            CognitiveMove.COMPARE: "Compares {option_a} vs {option_b}",
            CognitiveMove.SIMULATE: "Models scenario: {scenario}",
            CognitiveMove.DECIDE: "Decides between {options}",
            CognitiveMove.COMMIT: "Commits to action: {action}",
        }
        return templates.get(move, "Adds cognitive nuance: {nuance}")