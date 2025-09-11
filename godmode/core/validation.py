"""
Validation engine for the GodMode reasoning system.

This module provides comprehensive validation and quality assessment
for reasoning processes and solutions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from godmode.models.core import (
    Solution,
    ReasoningTrace,
    CognitiveState,
    Problem,
    ReasoningStep,
)


logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for reasoning assessment."""
    coherence: float = 0.0
    consistency: float = 0.0
    completeness: float = 0.0
    feasibility: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    overall_score: float = 0.0


class ValidationEngine:
    """
    Validation engine for assessing reasoning quality and solution validity.
    """
    
    def __init__(
        self,
        coherence_weight: float = 0.2,
        consistency_weight: float = 0.2,
        completeness_weight: float = 0.2,
        feasibility_weight: float = 0.2,
        efficiency_weight: float = 0.1,
        novelty_weight: float = 0.1,
    ):
        """Initialize validation engine with quality weights."""
        self.weights = {
            'coherence': coherence_weight,
            'consistency': consistency_weight,
            'completeness': completeness_weight,
            'feasibility': feasibility_weight,
            'efficiency': efficiency_weight,
            'novelty': novelty_weight,
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Validation history for learning
        self.validation_history: List[Dict[str, Any]] = []
        
    def validate_solution(
        self,
        solution: Solution,
        problem: Problem,
        reasoning_trace: Optional[ReasoningTrace] = None,
    ) -> QualityMetrics:
        """
        Validate a solution and compute quality metrics.
        
        Args:
            solution: Solution to validate
            problem: Original problem
            reasoning_trace: Optional reasoning trace
            
        Returns:
            QualityMetrics with computed scores
        """
        metrics = QualityMetrics()
        
        # Assess coherence
        metrics.coherence = self._assess_coherence(solution, problem, reasoning_trace)
        
        # Assess consistency
        metrics.consistency = self._assess_consistency(solution, problem, reasoning_trace)
        
        # Assess completeness
        metrics.completeness = self._assess_completeness(solution, problem)
        
        # Assess feasibility
        metrics.feasibility = self._assess_feasibility(solution, problem)
        
        # Assess efficiency
        metrics.efficiency = self._assess_efficiency(solution, reasoning_trace)
        
        # Assess novelty
        metrics.novelty = self._assess_novelty(solution, problem)
        
        # Compute overall score
        metrics.overall_score = self._compute_overall_score(metrics)
        
        # Store validation result
        self.validation_history.append({
            'problem_id': str(problem.id),
            'solution_id': str(solution.id),
            'metrics': metrics,
            'timestamp': solution.created_at,
        })
        
        logger.info(f"Solution validation complete. Overall score: {metrics.overall_score:.3f}")
        
        return metrics
    
    def validate_reasoning_trace(
        self,
        reasoning_trace: ReasoningTrace,
        problem: Problem,
    ) -> Dict[str, float]:
        """
        Validate a reasoning trace and compute process quality metrics.
        
        Args:
            reasoning_trace: Reasoning trace to validate
            problem: Original problem
            
        Returns:
            Dict with process quality scores
        """
        metrics = {}
        
        # Logical consistency across steps
        metrics['logical_consistency'] = self._assess_logical_consistency(reasoning_trace)
        
        # Step coherence
        metrics['step_coherence'] = self._assess_step_coherence(reasoning_trace)
        
        # Progress towards goal
        metrics['goal_progress'] = self._assess_goal_progress(reasoning_trace, problem)
        
        # Resource efficiency
        metrics['resource_efficiency'] = self._assess_resource_efficiency(reasoning_trace)
        
        # Cognitive load balance
        metrics['cognitive_balance'] = self._assess_cognitive_balance(reasoning_trace)
        
        return metrics
    
    def _assess_coherence(
        self,
        solution: Solution,
        problem: Problem,
        reasoning_trace: Optional[ReasoningTrace],
    ) -> float:
        """Assess solution coherence."""
        score = 0.0
        
        # Check if solution addresses the problem
        problem_keywords = set(problem.description.lower().split())
        solution_keywords = set(solution.solution_text.lower().split())
        
        keyword_overlap = len(problem_keywords & solution_keywords)
        keyword_score = min(1.0, keyword_overlap / max(1, len(problem_keywords) * 0.3))
        score += keyword_score * 0.4
        
        # Check solution structure
        if len(solution.solution_text.split()) > 10:  # Reasonable length
            score += 0.3
        
        if any(word in solution.solution_text.lower() for word in ['because', 'therefore', 'thus', 'since']):
            score += 0.3  # Contains reasoning indicators
        
        # Use reasoning trace if available
        if reasoning_trace and reasoning_trace.coherence > 0:
            score = (score + reasoning_trace.coherence) / 2
        
        return min(1.0, score)
    
    def _assess_consistency(
        self,
        solution: Solution,
        problem: Problem,
        reasoning_trace: Optional[ReasoningTrace],
    ) -> float:
        """Assess solution consistency."""
        score = 0.7  # Base consistency score
        
        # Check for contradictions in solution text
        contradiction_indicators = ['but', 'however', 'although', 'despite', 'nevertheless']
        contradiction_count = sum(1 for word in contradiction_indicators 
                                if word in solution.solution_text.lower())
        
        # Some contradictions might be acceptable for complex problems
        if contradiction_count > 3:
            score -= 0.2
        
        # Check consistency with problem constraints
        for constraint in problem.constraints:
            if any(word in solution.solution_text.lower() 
                  for word in constraint.lower().split()):
                score += 0.1
        
        # Use reasoning trace consistency if available
        if reasoning_trace and reasoning_trace.consistency > 0:
            score = (score + reasoning_trace.consistency) / 2
        
        return min(1.0, max(0.0, score))
    
    def _assess_completeness(self, solution: Solution, problem: Problem) -> float:
        """Assess solution completeness."""
        score = 0.0
        
        # Check if solution addresses all objectives
        objectives_addressed = 0
        for objective in problem.objectives:
            if any(word in solution.solution_text.lower() 
                  for word in objective.lower().split()):
                objectives_addressed += 1
        
        if problem.objectives:
            score += (objectives_addressed / len(problem.objectives)) * 0.5
        else:
            score += 0.5  # No specific objectives
        
        # Check solution length relative to problem complexity
        problem_complexity = len(problem.description.split())
        solution_length = len(solution.solution_text.split())
        
        if solution_length >= problem_complexity * 0.5:
            score += 0.3
        
        # Check for implementation details
        implementation_indicators = ['step', 'first', 'then', 'next', 'finally', 'process']
        if any(word in solution.solution_text.lower() for word in implementation_indicators):
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_feasibility(self, solution: Solution, problem: Problem) -> float:
        """Assess solution feasibility."""
        score = 0.6  # Base feasibility score
        
        # Check for unrealistic claims
        unrealistic_indicators = ['impossible', 'never', 'always', '100%', 'perfect', 'infinite']
        unrealistic_count = sum(1 for word in unrealistic_indicators 
                              if word in solution.solution_text.lower())
        
        score -= unrealistic_count * 0.1
        
        # Check for practical considerations
        practical_indicators = ['cost', 'time', 'resource', 'budget', 'realistic', 'achievable']
        practical_count = sum(1 for word in practical_indicators 
                            if word in solution.solution_text.lower())
        
        score += min(0.3, practical_count * 0.1)
        
        # Check against problem constraints
        constraint_violations = 0
        for constraint in problem.constraints:
            # Simple heuristic: if solution mentions opposite of constraint
            constraint_words = constraint.lower().split()
            if any(f"not {word}" in solution.solution_text.lower() 
                  or f"no {word}" in solution.solution_text.lower()
                  for word in constraint_words):
                constraint_violations += 1
        
        score -= constraint_violations * 0.2
        
        return min(1.0, max(0.0, score))
    
    def _assess_efficiency(
        self,
        solution: Solution,
        reasoning_trace: Optional[ReasoningTrace],
    ) -> float:
        """Assess solution efficiency."""
        score = 0.5  # Base efficiency score
        
        # Check solution conciseness
        solution_length = len(solution.solution_text.split())
        if solution_length < 50:  # Concise solution
            score += 0.2
        elif solution_length > 200:  # Very verbose
            score -= 0.1
        
        # Use reasoning trace efficiency if available
        if reasoning_trace:
            if reasoning_trace.efficiency > 0:
                score = (score + reasoning_trace.efficiency) / 2
            
            # Penalize excessive reasoning steps
            if len(reasoning_trace.steps) > 50:
                score -= 0.2
        
        # Check for efficiency-related terms
        efficiency_indicators = ['efficient', 'optimal', 'minimize', 'maximize', 'streamline']
        if any(word in solution.solution_text.lower() for word in efficiency_indicators):
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _assess_novelty(self, solution: Solution, problem: Problem) -> float:
        """Assess solution novelty."""
        score = 0.5  # Base novelty score
        
        # Check for creative indicators
        creative_indicators = ['innovative', 'novel', 'creative', 'unique', 'original', 'breakthrough']
        creative_count = sum(1 for word in creative_indicators 
                           if word in solution.solution_text.lower())
        
        score += min(0.3, creative_count * 0.15)
        
        # Check against common solutions (simplified heuristic)
        common_solutions = ['increase', 'decrease', 'improve', 'reduce', 'enhance', 'optimize']
        common_count = sum(1 for word in common_solutions 
                         if word in solution.solution_text.lower())
        
        # Too many common words might indicate lack of novelty
        if common_count > 5:
            score -= 0.2
        
        # Compare with historical solutions (if available)
        if len(self.validation_history) > 0:
            # Simple similarity check with recent solutions
            recent_solutions = [h for h in self.validation_history[-10:]]
            similarity_scores = []
            
            for hist_entry in recent_solutions:
                # Simplified text similarity (would use embeddings in practice)
                hist_words = set(hist_entry.get('solution_text', '').lower().split())
                current_words = set(solution.solution_text.lower().split())
                
                if hist_words and current_words:
                    similarity = len(hist_words & current_words) / len(hist_words | current_words)
                    similarity_scores.append(similarity)
            
            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                score -= avg_similarity * 0.3  # Reduce novelty if too similar
        
        return min(1.0, max(0.0, score))
    
    def _compute_overall_score(self, metrics: QualityMetrics) -> float:
        """Compute overall quality score from individual metrics."""
        score = (
            metrics.coherence * self.weights['coherence'] +
            metrics.consistency * self.weights['consistency'] +
            metrics.completeness * self.weights['completeness'] +
            metrics.feasibility * self.weights['feasibility'] +
            metrics.efficiency * self.weights['efficiency'] +
            metrics.novelty * self.weights['novelty']
        )
        
        return min(1.0, max(0.0, score))
    
    def _assess_logical_consistency(self, reasoning_trace: ReasoningTrace) -> float:
        """Assess logical consistency across reasoning steps."""
        if not reasoning_trace.steps:
            return 0.5
        
        # Check for logical flow between steps
        consistency_score = 0.0
        
        for i in range(1, len(reasoning_trace.steps)):
            prev_step = reasoning_trace.steps[i-1]
            curr_step = reasoning_trace.steps[i]
            
            # Check if current step builds on previous step
            prev_outputs = set(str(prev_step.output_state).lower().split())
            curr_inputs = set(str(curr_step.input_state).lower().split())
            
            overlap = len(prev_outputs & curr_inputs)
            if overlap > 0:
                consistency_score += 1.0
        
        if len(reasoning_trace.steps) > 1:
            consistency_score /= (len(reasoning_trace.steps) - 1)
        else:
            consistency_score = 1.0
        
        return min(1.0, consistency_score)
    
    def _assess_step_coherence(self, reasoning_trace: ReasoningTrace) -> float:
        """Assess coherence within individual reasoning steps."""
        if not reasoning_trace.steps:
            return 0.5
        
        coherence_scores = []
        
        for step in reasoning_trace.steps:
            step_score = 0.5  # Base score
            
            # Check if step has rationale
            if step.rationale:
                step_score += 0.3
            
            # Check confidence level
            step_score += step.confidence * 0.2
            
            coherence_scores.append(step_score)
        
        return sum(coherence_scores) / len(coherence_scores)
    
    def _assess_goal_progress(self, reasoning_trace: ReasoningTrace, problem: Problem) -> float:
        """Assess progress towards problem-solving goals."""
        if not reasoning_trace.steps:
            return 0.0
        
        # Check if reasoning moves towards problem objectives
        progress_score = 0.0
        
        problem_keywords = set(problem.description.lower().split())
        
        for i, step in enumerate(reasoning_trace.steps):
            step_text = f"{step.operation} {step.rationale or ''}"
            step_keywords = set(step_text.lower().split())
            
            # Check relevance to problem
            relevance = len(problem_keywords & step_keywords) / max(1, len(problem_keywords))
            
            # Weight later steps more (should be more focused)
            weight = (i + 1) / len(reasoning_trace.steps)
            
            progress_score += relevance * weight
        
        return min(1.0, progress_score)
    
    def _assess_resource_efficiency(self, reasoning_trace: ReasoningTrace) -> float:
        """Assess efficiency of resource usage in reasoning."""
        if not reasoning_trace.steps:
            return 0.5
        
        efficiency_score = 1.0
        
        # Penalize excessive steps
        if len(reasoning_trace.steps) > 20:
            efficiency_score -= 0.3
        
        # Penalize excessive time
        if reasoning_trace.total_time and reasoning_trace.total_time > 60:  # More than 1 minute
            efficiency_score -= 0.2
        
        # Reward high-confidence steps
        if reasoning_trace.steps:
            avg_confidence = sum(step.confidence for step in reasoning_trace.steps) / len(reasoning_trace.steps)
            efficiency_score += (avg_confidence - 0.5) * 0.2
        
        return min(1.0, max(0.0, efficiency_score))
    
    def _assess_cognitive_balance(self, reasoning_trace: ReasoningTrace) -> float:
        """Assess balance across cognitive levels."""
        if not reasoning_trace.hierarchical_levels:
            return 0.5
        
        # Check if multiple cognitive levels were used
        levels_used = len(reasoning_trace.hierarchical_levels)
        balance_score = min(1.0, levels_used / 4.0)  # Ideal is using all 4 levels
        
        # Check distribution of steps across levels
        step_counts = [len(steps) for steps in reasoning_trace.hierarchical_levels.values()]
        if step_counts:
            # Calculate coefficient of variation (lower is more balanced)
            mean_steps = sum(step_counts) / len(step_counts)
            if mean_steps > 0:
                std_steps = np.std(step_counts)
                cv = std_steps / mean_steps
                balance_score += max(0, 0.5 - cv)  # Reward balanced distribution
        
        return min(1.0, balance_score)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics from validation history."""
        if not self.validation_history:
            return {"total_validations": 0}
        
        recent_metrics = [entry['metrics'] for entry in self.validation_history[-100:]]
        
        stats = {
            "total_validations": len(self.validation_history),
            "average_coherence": np.mean([m.coherence for m in recent_metrics]),
            "average_consistency": np.mean([m.consistency for m in recent_metrics]),
            "average_completeness": np.mean([m.completeness for m in recent_metrics]),
            "average_feasibility": np.mean([m.feasibility for m in recent_metrics]),
            "average_efficiency": np.mean([m.efficiency for m in recent_metrics]),
            "average_novelty": np.mean([m.novelty for m in recent_metrics]),
            "average_overall_score": np.mean([m.overall_score for m in recent_metrics]),
        }
        
        return stats