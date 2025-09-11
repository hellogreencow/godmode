"""Backward reasoning for generating PRIOR ladders."""

import asyncio
import re
from typing import List, Optional, Dict, Any, Tuple
from ...models.core import Question, CognitiveMove
from ...models.commands import Budgets
from .cognitive_moves import CognitiveMoveProgression
from .scoring import ScoreCalculator


class BackwardReasoning:
    """
    Generates PRIOR ladders through backward reasoning.
    
    Creates minimal coverage set of questions that would collapse 
    the current question if answered in order.
    """
    
    def __init__(self):
        self.move_progression = CognitiveMoveProgression()
        self.scorer = ScoreCalculator()
        self._question_counter = 0
    
    async def enumerate_priors(
        self, 
        current_question: str, 
        context: Optional[str],
        budgets: Budgets
    ) -> List[Question]:
        """
        Phase 1: ENUMERATE - Generate diverse prior question candidates.
        
        Uses premise mining to extract hidden assumptions and generate
        seed ladders at levels 1-2 using define/scope moves.
        """
        # Extract premises and assumptions
        premises = await self._extract_premises(current_question, context)
        
        # Generate seed questions for each premise
        seed_tasks = [
            self._generate_seed_ladder(premise, budgets)
            for premise in premises[:budgets.beam_width]
        ]
        
        seed_ladders = await asyncio.gather(*seed_tasks)
        
        # Flatten and return candidates
        candidates = []
        for ladder in seed_ladders:
            candidates.extend(ladder)
        
        return candidates
    
    async def rerank_candidates(self, candidates: List[Question]) -> List[Question]:
        """
        Phase 2: RERANK - Score candidates by expected_info_gain × coherence × effort.
        """
        # Calculate scores for all candidates
        scored_candidates = []
        
        for candidate in candidates:
            info_gain = self.scorer.calculate_info_gain(candidate)
            coherence = self.scorer.calculate_coherence(candidate, candidates)
            effort_penalty = self.scorer.calculate_effort_penalty(candidate)
            
            rank_score = info_gain * coherence / (1 + effort_penalty)
            
            # Update candidate with scores
            candidate.expected_info_gain = info_gain
            candidate.confidence = self.scorer.calculate_confidence(candidate)
            
            scored_candidates.append((candidate, rank_score))
        
        # Sort by rank score and return top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in scored_candidates]
    
    async def stitch_ladder(self, ranked_candidates: List[Question]) -> List[Question]:
        """
        Phase 3: STITCH - Wire builds_on relationships and create coherent ladder.
        """
        if not ranked_candidates:
            return []
        
        # Build dependency graph
        stitched_ladder = []
        level_map = {}  # level -> [questions]
        
        for candidate in ranked_candidates:
            level = candidate.level
            if level not in level_map:
                level_map[level] = []
            level_map[level].append(candidate)
        
        # Process levels in order
        for level in sorted(level_map.keys()):
            questions_at_level = level_map[level]
            
            if level == 1:
                # Level 1 questions have no dependencies
                for q in questions_at_level:
                    q.builds_on = []
                    stitched_ladder.append(q)
            else:
                # Higher level questions build on previous levels
                for q in questions_at_level:
                    # Find appropriate parents from previous levels
                    parents = self._find_logical_parents(q, stitched_ladder)
                    q.builds_on = [p.id for p in parents]
                    
                    # Only add if it has valid parents
                    if q.builds_on:
                        stitched_ladder.append(q)
        
        return stitched_ladder
    
    async def _extract_premises(self, question: str, context: Optional[str]) -> List[str]:
        """Extract hidden premises and assumptions from the question."""
        premises = []
        
        # Rule-based premise extraction patterns
        premise_patterns = [
            # Assumptions about definitions
            (r'\b(should|must|need to|have to)\b', "definition_of_necessity"),
            (r'\b(best|optimal|ideal)\b', "definition_of_optimality"),
            (r'\b(success|failure|good|bad)\b', "definition_of_success_criteria"),
            
            # Assumptions about scope
            (r'\b(when|where|how long|how much)\b', "scope_constraints"),
            (r'\b(all|every|some|most)\b', "scope_quantifiers"),
            
            # Assumptions about comparison
            (r'\b(better than|worse than|compared to)\b', "comparison_baseline"),
            (r'\b(more|less|higher|lower)\b', "comparison_metrics"),
            
            # Assumptions about causation
            (r'\b(because|since|due to|caused by)\b', "causal_assumptions"),
            (r'\b(if|when|unless)\b', "conditional_assumptions"),
        ]
        
        question_lower = question.lower()
        
        for pattern, premise_type in premise_patterns:
            if re.search(pattern, question_lower):
                premises.append(self._generate_premise_question(premise_type, question))
        
        # Always include fundamental premises
        premises.extend([
            f"What does 'success' mean in the context of: {question}?",
            f"What are the key constraints and boundaries for: {question}?",
            f"What metrics would help evaluate options for: {question}?",
        ])
        
        return premises[:6]  # Limit to avoid explosion
    
    def _generate_premise_question(self, premise_type: str, original_question: str) -> str:
        """Generate a specific premise question based on type."""
        premise_templates = {
            "definition_of_necessity": "What exactly needs to be achieved, and why is it necessary?",
            "definition_of_optimality": "How do we define 'optimal' or 'best' in this context?",
            "definition_of_success_criteria": "What specific criteria define success vs failure here?",
            "scope_constraints": "What are the time, resource, and situational constraints?",
            "scope_quantifiers": "What is the exact scope - who/what/when/where does this apply to?",
            "comparison_baseline": "What is the baseline or reference point for comparison?",
            "comparison_metrics": "What metrics or dimensions should be used for comparison?",
            "causal_assumptions": "What causal relationships are assumed to be true?",
            "conditional_assumptions": "What conditions or prerequisites are assumed?",
        }
        
        return premise_templates.get(premise_type, "What assumptions underlie this question?")
    
    async def _generate_seed_ladder(self, premise: str, budgets: Budgets) -> List[Question]:
        """Generate a seed ladder from a premise using cognitive moves."""
        ladder = []
        
        # Level 1: Define (premise as-is)
        q1 = Question(
            id=f"QP{self._next_id()}",
            text=premise,
            level=1,
            cognitive_move=CognitiveMove.DEFINE,
            builds_on=[],
            delta_nuance="Establishes fundamental definition or constraint",
            expected_info_gain=0.0,  # Will be calculated in rerank
            confidence=0.0,  # Will be calculated in rerank
            tags=["premise", "definition"]
        )
        ladder.append(q1)
        
        # Level 2: Scope (add boundaries)
        scope_question = self._generate_scope_question(premise)
        q2 = Question(
            id=f"QP{self._next_id()}",
            text=scope_question,
            level=2,
            cognitive_move=CognitiveMove.SCOPE,
            builds_on=[q1.id],
            delta_nuance="Adds scope boundaries and constraints",
            expected_info_gain=0.0,
            confidence=0.0,
            tags=["premise", "scope"]
        )
        ladder.append(q2)
        
        # Optionally add level 3: Quantify
        if budgets.depth_back >= 3:
            quantify_question = self._generate_quantify_question(premise)
            q3 = Question(
                id=f"QP{self._next_id()}",
                text=quantify_question,
                level=3,
                cognitive_move=CognitiveMove.QUANTIFY,
                builds_on=[q2.id],
                delta_nuance="Adds quantitative metrics and thresholds",
                expected_info_gain=0.0,
                confidence=0.0,
                tags=["premise", "quantify"]
            )
            ladder.append(q3)
        
        return ladder
    
    def _generate_scope_question(self, premise: str) -> str:
        """Generate a scoping question from a premise."""
        return f"Within what boundaries and constraints should we consider: {premise}"
    
    def _generate_quantify_question(self, premise: str) -> str:
        """Generate a quantification question from a premise."""
        return f"How can we measure and quantify progress on: {premise}"
    
    def _find_logical_parents(self, question: Question, previous_questions: List[Question]) -> List[Question]:
        """Find logical parent questions for builds_on relationships."""
        parents = []
        
        # Find questions at lower levels that this could logically build upon
        for prev_q in previous_questions:
            if prev_q.level < question.level:
                # Simple heuristic: check for keyword overlap and cognitive move progression
                if self._has_logical_dependency(question, prev_q):
                    parents.append(prev_q)
        
        # If no parents found, connect to highest-level available question
        if not parents and previous_questions:
            highest_level_questions = [q for q in previous_questions if q.level == max(pq.level for pq in previous_questions)]
            parents = highest_level_questions[:1]
        
        return parents
    
    def _has_logical_dependency(self, question: Question, potential_parent: Question) -> bool:
        """Check if question logically depends on potential parent."""
        # Check cognitive move progression
        if not self.move_progression.is_valid_progression(potential_parent.cognitive_move, question.cognitive_move):
            return False
        
        # Check for keyword/concept overlap
        q_words = set(question.text.lower().split())
        p_words = set(potential_parent.text.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "what", "how", "why", "when", "where"}
        q_words -= stop_words
        p_words -= stop_words
        
        # Check for meaningful overlap
        overlap = len(q_words & p_words)
        return overlap >= 2 or len(q_words & p_words) / max(len(q_words), len(p_words), 1) > 0.3
    
    def _next_id(self) -> int:
        """Get next question ID number."""
        self._question_counter += 1
        return self._question_counter