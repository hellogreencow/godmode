"""Scoring algorithms for question ranking and confidence calibration."""

import math
import re
from typing import List, Dict, Set
from ...models.core import Question, CognitiveMove


class ScoreCalculator:
    """
    Calculates scoring metrics for question ranking and confidence calibration.
    
    Core scoring formula: rank_score = info_gain * coherence / (1 + effort_penalty)
    """
    
    def __init__(self):
        # Cognitive move weights for info gain calculation
        self.move_weights = {
            CognitiveMove.DEFINE: 0.9,      # High info gain - establishes foundation
            CognitiveMove.SCOPE: 0.8,       # High info gain - reduces uncertainty space
            CognitiveMove.QUANTIFY: 0.7,    # Good info gain - adds measurability
            CognitiveMove.COMPARE: 0.6,     # Moderate info gain - narrows options
            CognitiveMove.SIMULATE: 0.5,    # Moderate info gain - tests scenarios
            CognitiveMove.DECIDE: 0.4,      # Lower info gain - applies knowledge
            CognitiveMove.COMMIT: 0.2,      # Lowest info gain - execution focused
        }
        
        # Effort penalty factors
        self.effort_factors = {
            "complexity_keywords": ["analyze", "evaluate", "research", "investigate", "study"],
            "time_keywords": ["long-term", "ongoing", "continuous", "extended"],
            "resource_keywords": ["expensive", "costly", "requires", "needs", "budget"],
        }
    
    def calculate_info_gain(self, question: Question) -> float:
        """
        Calculate expected information gain (0.0 to 1.0).
        
        Based on:
        - Cognitive move type (foundational moves have higher gain)
        - Question specificity (more specific = higher gain)
        - Uncertainty reduction potential
        """
        # Base score from cognitive move
        base_score = self.move_weights.get(question.cognitive_move, 0.5)
        
        # Specificity bonus
        specificity_score = self._calculate_specificity(question.text)
        
        # Level penalty (higher levels have diminishing returns)
        level_penalty = 1.0 / (1.0 + 0.1 * (question.level - 1))
        
        # Combine factors
        info_gain = base_score * (0.7 + 0.3 * specificity_score) * level_penalty
        
        return min(1.0, max(0.0, info_gain))
    
    def calculate_coherence(self, question: Question, all_questions: List[Question]) -> float:
        """
        Calculate coherence with existing questions (0.0 to 1.0).
        
        Based on:
        - Semantic similarity with related questions
        - Logical flow with parent questions
        - Consistency with cognitive move progression
        """
        if not all_questions:
            return 1.0
        
        # Find related questions (same level or parents)
        related_questions = [
            q for q in all_questions 
            if q.id != question.id and (
                q.level == question.level or 
                q.id in question.builds_on or
                question.id in q.builds_on
            )
        ]
        
        if not related_questions:
            return 1.0
        
        # Calculate semantic coherence
        semantic_scores = []
        for related in related_questions:
            similarity = self._calculate_semantic_similarity(question.text, related.text)
            semantic_scores.append(similarity)
        
        avg_semantic = sum(semantic_scores) / len(semantic_scores)
        
        # Calculate logical flow coherence
        logical_coherence = self._calculate_logical_coherence(question, all_questions)
        
        # Combine factors
        coherence = 0.6 * avg_semantic + 0.4 * logical_coherence
        
        return min(1.0, max(0.0, coherence))
    
    def calculate_effort_penalty(self, question: Question) -> float:
        """
        Calculate effort penalty (0.0 = no penalty, higher = more effort).
        
        Based on:
        - Question complexity indicators
        - Time requirements
        - Resource requirements
        """
        text_lower = question.text.lower()
        penalty = 0.0
        
        # Complexity penalty
        complexity_count = sum(1 for keyword in self.effort_factors["complexity_keywords"] 
                             if keyword in text_lower)
        penalty += complexity_count * 0.2
        
        # Time penalty
        time_count = sum(1 for keyword in self.effort_factors["time_keywords"] 
                        if keyword in text_lower)
        penalty += time_count * 0.3
        
        # Resource penalty
        resource_count = sum(1 for keyword in self.effort_factors["resource_keywords"] 
                           if keyword in text_lower)
        penalty += resource_count * 0.25
        
        # Question length penalty (longer questions often more complex)
        length_penalty = min(0.3, len(question.text.split()) / 100.0)
        penalty += length_penalty
        
        return penalty
    
    def calculate_confidence(self, question: Question) -> float:
        """
        Calculate confidence that user will want this question next (0.0 to 1.0).
        
        Based on:
        - Information gain potential
        - Question clarity and actionability
        - Position in cognitive progression
        """
        # Base confidence from info gain
        base_confidence = question.expected_info_gain
        
        # Clarity bonus
        clarity_score = self._calculate_clarity(question.text)
        
        # Cognitive move appropriateness
        move_appropriateness = self._calculate_move_appropriateness(question)
        
        # Combine factors
        confidence = base_confidence * (0.5 + 0.3 * clarity_score + 0.2 * move_appropriateness)
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_specificity(self, text: str) -> float:
        """Calculate how specific/concrete a question is."""
        # Count specific indicators
        specific_indicators = [
            r'\b\d+\b',  # Numbers
            r'\b(who|what|when|where|how much|how many)\b',  # Specific question words
            r'\b(exactly|specifically|precisely)\b',  # Precision words
            r'\b[A-Z][a-z]+\b',  # Proper nouns
        ]
        
        specificity_count = 0
        for pattern in specific_indicators:
            specificity_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Normalize by text length
        words = len(text.split())
        specificity_ratio = specificity_count / max(words, 1)
        
        return min(1.0, specificity_ratio * 3)  # Scale up the ratio
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (simple implementation)."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "what", "how", "why", "when", "where"}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_logical_coherence(self, question: Question, all_questions: List[Question]) -> float:
        """Calculate logical coherence with parent questions."""
        if not question.builds_on:
            return 1.0  # Root questions are inherently coherent
        
        coherence_scores = []
        
        for parent_id in question.builds_on:
            parent = next((q for q in all_questions if q.id == parent_id), None)
            if parent:
                # Check cognitive move progression
                move_coherence = 1.0 if self._is_valid_move_progression(parent.cognitive_move, question.cognitive_move) else 0.3
                
                # Check semantic relationship
                semantic_coherence = self._calculate_semantic_similarity(parent.text, question.text)
                
                # Check level progression
                level_coherence = 1.0 if question.level > parent.level else 0.5
                
                parent_coherence = (move_coherence + semantic_coherence + level_coherence) / 3
                coherence_scores.append(parent_coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0
    
    def _calculate_clarity(self, text: str) -> float:
        """Calculate how clear and actionable a question is."""
        clarity_indicators = [
            r'\b(what|how|why|when|where|which)\b',  # Clear question words
            r'\?',  # Question marks
            r'\b(define|identify|determine|calculate|measure)\b',  # Action verbs
        ]
        
        clarity_count = 0
        for pattern in clarity_indicators:
            clarity_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Penalty for vague words
        vague_words = ["thing", "stuff", "something", "somehow", "maybe", "perhaps", "might"]
        vague_count = sum(1 for word in vague_words if word in text.lower())
        
        clarity_score = clarity_count / max(len(text.split()), 1) - vague_count * 0.1
        
        return min(1.0, max(0.0, clarity_score))
    
    def _calculate_move_appropriateness(self, question: Question) -> float:
        """Calculate how appropriate the cognitive move is for this question."""
        # Simple heuristic based on question content
        text_lower = question.text.lower()
        
        move_indicators = {
            CognitiveMove.DEFINE: ["what is", "define", "meaning", "definition"],
            CognitiveMove.SCOPE: ["within", "boundary", "limit", "constraint", "scope"],
            CognitiveMove.QUANTIFY: ["how much", "how many", "measure", "metric", "number"],
            CognitiveMove.COMPARE: ["compare", "versus", "vs", "better", "worse", "difference"],
            CognitiveMove.SIMULATE: ["what if", "scenario", "model", "predict", "simulate"],
            CognitiveMove.DECIDE: ["should", "choose", "decide", "select", "pick"],
            CognitiveMove.COMMIT: ["will", "commit", "do", "implement", "execute"],
        }
        
        indicators = move_indicators.get(question.cognitive_move, [])
        indicator_count = sum(1 for indicator in indicators if indicator in text_lower)
        
        return min(1.0, indicator_count / max(len(indicators), 1))
    
    def _is_valid_move_progression(self, from_move: CognitiveMove, to_move: CognitiveMove) -> bool:
        """Check if cognitive move progression is valid (simplified)."""
        move_order = [
            CognitiveMove.DEFINE,
            CognitiveMove.SCOPE,
            CognitiveMove.QUANTIFY,
            CognitiveMove.COMPARE,
            CognitiveMove.SIMULATE,
            CognitiveMove.DECIDE,
            CognitiveMove.COMMIT,
        ]
        
        try:
            from_idx = move_order.index(from_move)
            to_idx = move_order.index(to_move)
            return to_idx >= from_idx  # Allow staying same or moving forward
        except ValueError:
            return True  # If moves not in standard order, assume valid