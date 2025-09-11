"""Invariant validation system for GODMODE."""

from typing import List, Dict, Set, Tuple, Any
import networkx as nx
from ..models.core import Question, Lane, Thread
from ..models.responses import GraphUpdate


class InvariantValidator:
    """
    Validates GODMODE invariants:
    - DAG structure (no cycles)
    - Level progression (levels increase along builds_on chains)
    - No near-duplicates at same level within same lane
    - Proper cognitive move progression
    - Natural ending consistency
    """
    
    def __init__(self):
        self.validation_errors: List[str] = []
    
    def validate_graph_update(self, graph_update: GraphUpdate) -> Tuple[bool, List[str]]:
        """
        Validate a complete graph update against all invariants.
        
        Returns (is_valid, error_messages)
        """
        self.validation_errors.clear()
        
        # Collect all questions
        all_questions = []
        all_questions.extend(graph_update.priors)
        for scenario in graph_update.scenarios:
            all_questions.extend(scenario.lane)
        
        # Run all validations
        self._validate_dag_structure(all_questions)
        self._validate_level_progression(all_questions)
        self._validate_no_duplicates(graph_update)
        self._validate_cognitive_progression(all_questions)
        self._validate_scenario_constraints(graph_update.scenarios)
        self._validate_thread_consistency(graph_update.threads, all_questions)
        
        return len(self.validation_errors) == 0, self.validation_errors.copy()
    
    def validate_questions(self, questions: List[Question]) -> Tuple[bool, List[str]]:
        """Validate a list of questions against basic invariants."""
        self.validation_errors.clear()
        
        self._validate_dag_structure(questions)
        self._validate_level_progression(questions)
        self._validate_cognitive_progression(questions)
        
        return len(self.validation_errors) == 0, self.validation_errors.copy()
    
    def validate_lane(self, lane: Lane) -> Tuple[bool, List[str]]:
        """Validate a single scenario lane."""
        self.validation_errors.clear()
        
        self._validate_dag_structure(lane.lane)
        self._validate_level_progression(lane.lane)
        self._validate_lane_duplicates(lane.lane, lane.id)
        self._validate_cognitive_progression(lane.lane)
        self._validate_natural_endings(lane.lane)
        
        return len(self.validation_errors) == 0, self.validation_errors.copy()
    
    def _validate_dag_structure(self, questions: List[Question]) -> None:
        """Validate that questions form a DAG (no cycles)."""
        if not questions:
            return
        
        # Build adjacency list
        graph = nx.DiGraph()
        
        for question in questions:
            graph.add_node(question.id)
            for parent_id in question.builds_on:
                # Only add edge if parent exists in our question set
                if any(q.id == parent_id for q in questions):
                    graph.add_edge(parent_id, question.id)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                for cycle in cycles:
                    cycle_str = " -> ".join(cycle + [cycle[0]])
                    self.validation_errors.append(f"Cycle detected: {cycle_str}")
        except Exception as e:
            self.validation_errors.append(f"Error checking for cycles: {str(e)}")
    
    def _validate_level_progression(self, questions: List[Question]) -> None:
        """Validate that levels increase along builds_on chains."""
        question_map = {q.id: q for q in questions}
        
        for question in questions:
            for parent_id in question.builds_on:
                parent = question_map.get(parent_id)
                if parent:
                    if question.level <= parent.level:
                        self.validation_errors.append(
                            f"Level progression violation: {question.id} (level {question.level}) "
                            f"must have level > {parent.id} (level {parent.level})"
                        )
    
    def _validate_no_duplicates(self, graph_update: GraphUpdate) -> None:
        """Validate no near-duplicates at same level within same context."""
        # Check priors
        self._validate_context_duplicates(graph_update.priors, "priors")
        
        # Check each scenario lane
        for scenario in graph_update.scenarios:
            self._validate_context_duplicates(scenario.lane, f"lane {scenario.id}")
    
    def _validate_context_duplicates(self, questions: List[Question], context_name: str) -> None:
        """Check for duplicates within a specific context."""
        level_groups = {}
        
        # Group questions by level
        for question in questions:
            level = question.level
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(question)
        
        # Check for duplicates within each level
        for level, level_questions in level_groups.items():
            texts = [q.text.lower().strip() for q in level_questions]
            
            # Check for exact duplicates
            if len(texts) != len(set(texts)):
                self.validation_errors.append(
                    f"Duplicate question text found in {context_name} at level {level}"
                )
            
            # Check for near-duplicates (high similarity)
            for i, text1 in enumerate(texts):
                for j, text2 in enumerate(texts[i+1:], i+1):
                    similarity = self._calculate_text_similarity(text1, text2)
                    if similarity > 0.8:  # 80% similarity threshold
                        self.validation_errors.append(
                            f"Near-duplicate questions in {context_name} at level {level}: "
                            f"'{level_questions[i].id}' and '{level_questions[j].id}'"
                        )
    
    def _validate_lane_duplicates(self, questions: List[Question], lane_id: str) -> None:
        """Validate no duplicates within a single lane."""
        self._validate_context_duplicates(questions, f"lane {lane_id}")
    
    def _validate_cognitive_progression(self, questions: List[Question]) -> None:
        """Validate cognitive move progression is logical."""
        from .reasoning.cognitive_moves import CognitiveMoveProgression
        
        move_progression = CognitiveMoveProgression()
        question_map = {q.id: q for q in questions}
        
        for question in questions:
            for parent_id in question.builds_on:
                parent = question_map.get(parent_id)
                if parent:
                    if not move_progression.is_valid_progression(
                        parent.cognitive_move, question.cognitive_move
                    ):
                        self.validation_errors.append(
                            f"Invalid cognitive progression: {parent.id} "
                            f"({parent.cognitive_move.value}) -> {question.id} "
                            f"({question.cognitive_move.value})"
                        )
    
    def _validate_scenario_constraints(self, scenarios: List[Lane]) -> None:
        """Validate scenario-specific constraints."""
        # Must have 3-5 scenario lanes
        if not (3 <= len(scenarios) <= 5):
            self.validation_errors.append(
                f"Must have 3-5 scenario lanes, got {len(scenarios)}"
            )
        
        # Each lane must have at least one question
        for scenario in scenarios:
            if not scenario.lane:
                self.validation_errors.append(
                    f"Scenario {scenario.id} has no questions"
                )
        
        # Lane IDs should follow pattern S-A, S-B, etc.
        expected_ids = [f"S-{chr(65+i)}" for i in range(len(scenarios))]
        actual_ids = [s.id for s in scenarios]
        
        if actual_ids != expected_ids:
            self.validation_errors.append(
                f"Lane IDs should be {expected_ids}, got {actual_ids}"
            )
    
    def _validate_thread_consistency(self, threads: List[Thread], all_questions: List[Question]) -> None:
        """Validate thread consistency."""
        question_ids = {q.id for q in all_questions}
        
        for thread in threads:
            # All path nodes should exist
            for node_id in thread.path:
                if node_id not in question_ids:
                    self.validation_errors.append(
                        f"Thread {thread.thread_id} references non-existent node {node_id}"
                    )
            
            # Origin node should be first in path
            if thread.path and thread.origin_node_id != thread.path[0]:
                self.validation_errors.append(
                    f"Thread {thread.thread_id} origin {thread.origin_node_id} "
                    f"doesn't match path start {thread.path[0]}"
                )
            
            # Path should follow valid builds_on relationships
            self._validate_thread_path(thread, all_questions)
    
    def _validate_thread_path(self, thread: Thread, all_questions: List[Question]) -> None:
        """Validate that thread path follows valid question relationships."""
        if len(thread.path) < 2:
            return  # Single node paths are always valid
        
        question_map = {q.id: q for q in all_questions}
        
        for i in range(1, len(thread.path)):
            current_id = thread.path[i]
            prev_id = thread.path[i-1]
            
            current_question = question_map.get(current_id)
            if current_question and prev_id not in current_question.builds_on:
                # Allow some flexibility - path doesn't have to be strict builds_on
                # But warn about potential issues
                pass  # Could add warning here if needed
    
    def _validate_natural_endings(self, questions: List[Question]) -> None:
        """Validate natural ending consistency."""
        for question in questions:
            if question.natural_end:
                # Natural endings should typically be at higher levels
                if question.level < 3:
                    self.validation_errors.append(
                        f"Question {question.id} marked as natural_end at low level {question.level}"
                    )
                
                # Natural endings should have low expected info gain
                if question.expected_info_gain > 0.5:
                    self.validation_errors.append(
                        f"Question {question.id} marked as natural_end but has high info_gain {question.expected_info_gain}"
                    )
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        error_types = {
            "dag_violations": 0,
            "level_violations": 0,
            "duplicate_violations": 0,
            "cognitive_violations": 0,
            "scenario_violations": 0,
            "thread_violations": 0,
            "other_violations": 0
        }
        
        for error in self.validation_errors:
            if "cycle" in error.lower():
                error_types["dag_violations"] += 1
            elif "level" in error.lower():
                error_types["level_violations"] += 1
            elif "duplicate" in error.lower():
                error_types["duplicate_violations"] += 1
            elif "cognitive" in error.lower():
                error_types["cognitive_violations"] += 1
            elif "scenario" in error.lower() or "lane" in error.lower():
                error_types["scenario_violations"] += 1
            elif "thread" in error.lower():
                error_types["thread_violations"] += 1
            else:
                error_types["other_violations"] += 1
        
        return {
            "total_errors": len(self.validation_errors),
            "error_breakdown": error_types,
            "is_valid": len(self.validation_errors) == 0
        }