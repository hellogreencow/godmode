"""
Model selection and management for GodMode system.

This module provides intelligent model selection based on problem characteristics
and user preferences.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from godmode.models.core import Problem
from .openrouter import OpenRouterIntegration, RECOMMENDED_MODELS


logger = logging.getLogger(__name__)


class ModelCategory(str, Enum):
    """Categories of models based on capabilities."""
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    FAST = "fast"
    COST_EFFECTIVE = "cost_effective"


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning."""
    model_id: str
    model_name: str
    category: ModelCategory
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_time: float


class ModelSelector:
    """Intelligent model selection for different problem types."""
    
    def __init__(self, openrouter: OpenRouterIntegration):
        self.openrouter = openrouter
        
        # Problem type to model category mapping
        self.problem_type_mapping = {
            "planning": ModelCategory.REASONING,
            "optimization": ModelCategory.ANALYTICAL,
            "design": ModelCategory.CREATIVE,
            "analysis": ModelCategory.ANALYTICAL,
            "creative": ModelCategory.CREATIVE,
            "general": ModelCategory.REASONING,
        }
        
        # Domain complexity scoring
        self.domain_complexity = {
            "artificial_intelligence": 0.9,
            "quantum_computing": 0.95,
            "biotechnology": 0.85,
            "finance": 0.8,
            "logistics": 0.7,
            "education": 0.6,
            "general": 0.5,
        }
    
    async def recommend_model(
        self,
        problem: Problem,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[ModelRecommendation]:
        """Recommend models for a given problem."""
        
        preferences = user_preferences or {}
        
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem(problem)
        
        # Get available models
        available_models = await self.openrouter.get_available_models()
        
        # Generate recommendations
        recommendations = []
        
        # Primary recommendation based on problem type
        primary_category = self.problem_type_mapping.get(
            problem.problem_type, 
            ModelCategory.REASONING
        )
        
        primary_models = RECOMMENDED_MODELS.get(primary_category.value, [])
        
        for model_id in primary_models[:3]:  # Top 3 from primary category
            model_info = self._find_model_info(model_id, available_models)
            if model_info:
                recommendation = ModelRecommendation(
                    model_id=model_id,
                    model_name=model_info.name,
                    category=primary_category,
                    confidence=self._calculate_confidence(problem_analysis, primary_category),
                    reasoning=self._generate_reasoning(problem_analysis, primary_category),
                    estimated_cost=self._estimate_cost(model_info, problem),
                    estimated_time=self._estimate_time(model_info, problem)
                )
                recommendations.append(recommendation)
        
        # Add fast option if problem is time-sensitive
        if problem_analysis.get("time_sensitive", False):
            fast_models = RECOMMENDED_MODELS.get("fast", [])
            for model_id in fast_models[:1]:  # Top fast model
                model_info = self._find_model_info(model_id, available_models)
                if model_info and model_id not in [r.model_id for r in recommendations]:
                    recommendation = ModelRecommendation(
                        model_id=model_id,
                        model_name=model_info.name,
                        category=ModelCategory.FAST,
                        confidence=0.7,
                        reasoning="Fast response for time-sensitive problem",
                        estimated_cost=self._estimate_cost(model_info, problem),
                        estimated_time=self._estimate_time(model_info, problem)
                    )
                    recommendations.append(recommendation)
        
        # Add cost-effective option if budget is a concern
        if preferences.get("budget_conscious", False):
            cost_effective_models = RECOMMENDED_MODELS.get("cost_effective", [])
            for model_id in cost_effective_models[:1]:  # Top cost-effective model
                model_info = self._find_model_info(model_id, available_models)
                if model_info and model_id not in [r.model_id for r in recommendations]:
                    recommendation = ModelRecommendation(
                        model_id=model_id,
                        model_name=model_info.name,
                        category=ModelCategory.COST_EFFECTIVE,
                        confidence=0.6,
                        reasoning="Cost-effective option for budget-conscious users",
                        estimated_cost=self._estimate_cost(model_info, problem),
                        estimated_time=self._estimate_time(model_info, problem)
                    )
                    recommendations.append(recommendation)
        
        # Sort by confidence and user preferences
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations
    
    def _analyze_problem(self, problem: Problem) -> Dict[str, Any]:
        """Analyze problem characteristics."""
        analysis = {}
        
        # Complexity analysis
        description_length = len(problem.description.split())
        constraint_count = len(problem.constraints)
        objective_count = len(problem.objectives)
        
        complexity_score = (
            min(1.0, description_length / 100) * 0.4 +
            min(1.0, constraint_count / 5) * 0.3 +
            min(1.0, objective_count / 3) * 0.3
        )
        
        domain_complexity = self.domain_complexity.get(problem.domain, 0.5)
        
        analysis["complexity"] = (complexity_score + domain_complexity) / 2
        analysis["description_length"] = description_length
        analysis["constraint_count"] = constraint_count
        analysis["objective_count"] = objective_count
        
        # Time sensitivity analysis
        time_keywords = ["urgent", "immediate", "asap", "quickly", "deadline", "time-sensitive"]
        analysis["time_sensitive"] = any(
            keyword in problem.description.lower() 
            for keyword in time_keywords
        )
        
        # Creativity requirements
        creative_keywords = ["creative", "innovative", "novel", "original", "breakthrough", "design"]
        analysis["requires_creativity"] = any(
            keyword in problem.description.lower() 
            for keyword in creative_keywords
        )
        
        # Analytical requirements
        analytical_keywords = ["analyze", "optimize", "calculate", "measure", "evaluate", "assess"]
        analysis["requires_analysis"] = any(
            keyword in problem.description.lower() 
            for keyword in analytical_keywords
        )
        
        return analysis
    
    def _calculate_confidence(
        self, 
        problem_analysis: Dict[str, Any], 
        category: ModelCategory
    ) -> float:
        """Calculate confidence for model recommendation."""
        base_confidence = 0.7
        
        # Adjust based on problem-category alignment
        if category == ModelCategory.CREATIVE and problem_analysis.get("requires_creativity", False):
            base_confidence += 0.2
        elif category == ModelCategory.ANALYTICAL and problem_analysis.get("requires_analysis", False):
            base_confidence += 0.2
        elif category == ModelCategory.FAST and problem_analysis.get("time_sensitive", False):
            base_confidence += 0.15
        
        # Adjust based on complexity
        complexity = problem_analysis.get("complexity", 0.5)
        if category in [ModelCategory.REASONING, ModelCategory.ANALYTICAL] and complexity > 0.7:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _generate_reasoning(
        self, 
        problem_analysis: Dict[str, Any], 
        category: ModelCategory
    ) -> str:
        """Generate reasoning for model recommendation."""
        reasons = []
        
        complexity = problem_analysis.get("complexity", 0.5)
        
        if category == ModelCategory.REASONING:
            reasons.append("Excellent for complex reasoning tasks")
            if complexity > 0.7:
                reasons.append("High complexity problem requires advanced reasoning")
        
        elif category == ModelCategory.CREATIVE:
            reasons.append("Optimized for creative and innovative solutions")
            if problem_analysis.get("requires_creativity", False):
                reasons.append("Problem explicitly requires creative thinking")
        
        elif category == ModelCategory.ANALYTICAL:
            reasons.append("Strong analytical and optimization capabilities")
            if problem_analysis.get("requires_analysis", False):
                reasons.append("Problem requires detailed analysis")
        
        elif category == ModelCategory.FAST:
            reasons.append("Fast response time for quick solutions")
            if problem_analysis.get("time_sensitive", False):
                reasons.append("Time-sensitive problem needs quick response")
        
        elif category == ModelCategory.COST_EFFECTIVE:
            reasons.append("Good balance of capability and cost")
        
        return "; ".join(reasons)
    
    def _find_model_info(self, model_id: str, available_models: List):
        """Find model info by ID."""
        for model in available_models:
            if model.id == model_id:
                return model
        return None
    
    def _estimate_cost(self, model_info, problem: Problem) -> float:
        """Estimate cost for processing the problem."""
        # Simplified cost estimation
        input_tokens = len(problem.description.split()) * 1.3  # Rough token estimate
        output_tokens = 500  # Estimated output length
        
        pricing = model_info.pricing
        
        if "prompt" in pricing and "completion" in pricing:
            input_cost = (input_tokens / 1000) * pricing["prompt"]
            output_cost = (output_tokens / 1000) * pricing["completion"]
            return input_cost + output_cost
        
        return 0.01  # Default estimate
    
    def _estimate_time(self, model_info, problem: Problem) -> float:
        """Estimate processing time in seconds."""
        # Simplified time estimation based on model and problem complexity
        base_time = 5.0  # Base processing time
        
        # Adjust based on model (some models are faster)
        if "gpt-3.5" in model_info.id or "claude-3-haiku" in model_info.id:
            base_time *= 0.5
        elif "gpt-4" in model_info.id or "claude-3.5" in model_info.id:
            base_time *= 1.5
        
        # Adjust based on problem complexity
        problem_length = len(problem.description.split())
        complexity_multiplier = min(2.0, problem_length / 100)
        
        return base_time * complexity_multiplier
