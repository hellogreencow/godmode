"""
OpenRouter API integration for GodMode system.

This module provides integration with OpenRouter's API to access
various large language models for enhanced reasoning capabilities.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests
import json
from pydantic import BaseModel, Field

from godmode.models.core import Problem, Solution


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    description: str
    context_length: int
    pricing: Dict[str, float]
    provider: str
    capabilities: List[str]


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter integration."""
    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field("https://openrouter.ai/api/v1", description="OpenRouter API base URL")
    default_model: str = Field("nvidia/nemotron-nano-9b-v2:free", description="Default model to use")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")


class OpenRouterIntegration:
    """Integration with OpenRouter API for accessing various LLMs."""

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """Initialize OpenRouter integration."""
        if config is None:
            # Try to get API key from environment
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                    "or provide OpenRouterConfig with api_key."
                )
            config = OpenRouterConfig(api_key=api_key)

        self.config = config

        # Use requests session exactly as per OpenRouter quickstart
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "HTTP-Referer": "https://github.com/godmode-ai/godmode",
            "X-Title": "GodMode Hierarchical Reasoning System",
        })

        # Available models cache
        self._models_cache: Optional[List[ModelInfo]] = None

    def _call_openrouter_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make OpenRouter API call using requests (exactly as per quickstart guide)."""
        try:
            # Use requests.post exactly as per OpenRouter quickstart
            response = self.session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload)  # Use data=json.dumps() as per quickstart
            )

            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and response_data["choices"]:
                return {
                    "success": True,
                    "content": response_data["choices"][0]["message"]["content"],
                    "model": payload.get("model", self.config.default_model),
                    "usage": response_data.get("usage", {}),
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                logger.error(f"No choices in response: {response_data}")
                return {
                    "success": False,
                    "error": f"No choices in response: {response_data}",
                    "model": payload.get("model", self.config.default_model)
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": payload.get("model", self.config.default_model)
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "success": False,
                "error": f"JSON decode error: {e}",
                "model": payload.get("model", self.config.default_model)
            }

    async def get_available_models(self, refresh: bool = False) -> List[ModelInfo]:
        """Get list of available models from OpenRouter."""
        if self._models_cache is None or refresh:
            try:
                response = await self.client.get("/models")
                response.raise_for_status()
                
                models_data = response.json()["data"]
                self._models_cache = [
                    ModelInfo(
                        id=model["id"],
                        name=model.get("name", model["id"]),
                        description=model.get("description", ""),
                        context_length=model.get("context_length", 4096),
                        pricing=model.get("pricing", {}),
                        provider=model.get("provider", "unknown"),
                        capabilities=model.get("capabilities", [])
                    )
                    for model in models_data
                ]
                
                logger.info(f"Loaded {len(self._models_cache)} models from OpenRouter")
                
            except Exception as e:
                logger.error(f"Failed to fetch models from OpenRouter: {e}")
                return []
        
        return self._models_cache or []
    
    def generate_reasoning(
        self,
        problem: Problem,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate reasoning for a problem using OpenRouter model."""

        model_id = model_id or self.config.default_model

        if system_prompt is None:
            system_prompt = self._create_reasoning_system_prompt()

        user_prompt = self._create_problem_prompt(problem)

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        # Use requests exactly as per OpenRouter quickstart guide
        logger.info(f"Making OpenRouter API call for model: {model_id}")
        return self._call_openrouter_api(payload)
    
    def solve_problem_with_model(
        self,
        problem: Problem,
        model_id: str,
        approach: str = "hierarchical"
    ) -> Solution:
        """Solve a problem using a specific OpenRouter model."""

        system_prompt = self._create_advanced_reasoning_prompt(approach)

        result = self.generate_reasoning(
            problem=problem,
            model_id=model_id,
            system_prompt=system_prompt,
            max_tokens=3000,
            temperature=0.3  # Lower temperature for more focused reasoning
        )
        
        if result["success"]:
            # Parse the response and create a Solution object
            solution = Solution(
                problem_id=problem.id,
                solution_text=result["content"],
                confidence=0.8,  # Could be enhanced with confidence parsing
                completeness=0.7,
                feasibility=0.8,
                solving_time=result.get("response_time", 0.0)
            )
            
            # Add metadata about the model used
            solution.solution_data = {
                "model_used": model_id,
                "approach": approach,
                "usage": result.get("usage", {}),
                "api_provider": "openrouter"
            }
            
            return solution
        else:
            # Create a failed solution
            return Solution(
                problem_id=problem.id,
                solution_text=f"Failed to generate solution: {result.get('error', 'Unknown error')}",
                confidence=0.0,
                completeness=0.0,
                feasibility=0.0
            )
    
    def _create_reasoning_system_prompt(self) -> str:
        """Create concise system prompt for reasoning tasks."""
        return """You are an AI assistant. Solve problems using structured reasoning.
Provide clear, logical solutions with confidence estimates."""
    
    def _create_problem_prompt(self, problem: Problem) -> str:
        """Create user prompt from problem."""
        prompt = f"PROBLEM: {problem.title}\n\n"
        prompt += f"DESCRIPTION: {problem.description}\n\n"
        
        if problem.constraints:
            prompt += f"CONSTRAINTS:\n"
            for constraint in problem.constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
        
        if problem.objectives:
            prompt += f"OBJECTIVES:\n"
            for objective in problem.objectives:
                prompt += f"- {objective}\n"
            prompt += "\n"
        
        prompt += f"DOMAIN: {problem.domain}\n"
        prompt += f"TYPE: {problem.problem_type}\n\n"
        
        prompt += "Please solve this problem using hierarchical reasoning. Show your thinking process at each cognitive level."
        
        return prompt
    
    def _create_advanced_reasoning_prompt(self, approach: str) -> str:
        """Create advanced system prompt based on approach."""
        base_prompt = self._create_reasoning_system_prompt()
        
        if approach == "forward":
            return base_prompt + "\n\nUse FORWARD REASONING: Start from known facts and work towards conclusions."
        elif approach == "backward":
            return base_prompt + "\n\nUse BACKWARD REASONING: Start from goals and work backwards to identify requirements."
        elif approach == "analogical":
            return base_prompt + "\n\nUse ANALOGICAL REASONING: Find similar problems and adapt their solutions."
        elif approach == "creative":
            return base_prompt + "\n\nUse CREATIVE REASONING: Think outside the box and generate novel approaches."
        else:
            return base_prompt + "\n\nUse HIERARCHICAL REASONING: Apply all cognitive levels systematically."
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Top recommended models for different use cases
RECOMMENDED_MODELS = {
    "reasoning": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4-turbo-preview",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.1-405b-instruct",
    ],
    "creative": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4-turbo-preview",
        "mistralai/mixtral-8x7b-instruct",
    ],
    "analytical": [
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-pro-1.5",
    ],
    "fast": [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku",
        "mistralai/mistral-7b-instruct",
    ],
    "cost_effective": [
        "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "anthropic/claude-3-haiku",
    ]
}
