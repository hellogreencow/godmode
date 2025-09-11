"""
Model routing and API interfaces for GODMODE
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import openai
import anthropic
import voyageai
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for model routing"""
    enumerate_model: str = os.getenv("ENUMERATE_MODEL", "gpt-3.5-turbo")
    stitch_model: str = os.getenv("STITCH_MODEL", "gpt-4-turbo-preview")
    rerank_model: str = os.getenv("RERANK_MODEL", "voyage-rerank-2.5")
    
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    voyage_key: str = os.getenv("VOYAGE_API_KEY", "")


class ModelInterface(ABC):
    """Abstract interface for model providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        pass


class OpenAIModel(ModelInterface):
    """OpenAI model interface"""
    
    def __init__(self, model_name: str, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
    
    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI generation failed: {str(e)}")


class AnthropicModel(ModelInterface):
    """Anthropic model interface"""
    
    def __init__(self, model_name: str, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model_name = model_name
    
    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic generation failed: {str(e)}")


class VoyageReranker:
    """Voyage AI reranking interface"""
    
    def __init__(self, api_key: str, model: str = "rerank-2.5"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
    
    async def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query
        Returns: List of (original_index, score) tuples sorted by relevance
        """
        try:
            # Voyage AI reranking is typically synchronous, wrap in asyncio
            def _sync_rerank():
                result = self.client.rerank(
                    query=query,
                    documents=documents,
                    model=self.model,
                    top_k=top_k or len(documents)
                )
                return [(r.index, r.relevance_score) for r in result.results]
            
            return await asyncio.get_event_loop().run_in_executor(None, _sync_rerank)
        except Exception as e:
            # Fallback to simple scoring if reranker fails
            return [(i, 0.5) for i in range(len(documents))]


class ModelRouter:
    """Routes requests to appropriate models based on task type"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.reranker = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize model clients based on configuration"""
        if self.config.openai_key:
            self.models["openai"] = {
                "gpt-3.5-turbo": OpenAIModel("gpt-3.5-turbo", self.config.openai_key),
                "gpt-4": OpenAIModel("gpt-4", self.config.openai_key),
                "gpt-4-turbo-preview": OpenAIModel("gpt-4-turbo-preview", self.config.openai_key),
            }
        
        if self.config.anthropic_key:
            self.models["anthropic"] = {
                "claude-3-haiku": AnthropicModel("claude-3-haiku-20240307", self.config.anthropic_key),
                "claude-3-sonnet": AnthropicModel("claude-3-sonnet-20240229", self.config.anthropic_key),
                "claude-3-opus": AnthropicModel("claude-3-opus-20240229", self.config.anthropic_key),
            }
        
        if self.config.voyage_key:
            self.reranker = VoyageReranker(self.config.voyage_key, "rerank-2.5")
    
    def get_model(self, model_name: str) -> ModelInterface:
        """Get model instance by name"""
        for provider_models in self.models.values():
            if model_name in provider_models:
                return provider_models[model_name]
        
        # Fallback to first available model
        for provider_models in self.models.values():
            return next(iter(provider_models.values()))
        
        raise Exception("No models available")
    
    async def enumerate(self, prompt: str, beam_width: int = 4) -> List[str]:
        """Generate diverse candidates using cheap model"""
        model = self.get_model(self.config.enumerate_model)
        
        # Generate multiple candidates with different temperatures
        tasks = []
        temperatures = [0.2, 0.7] * (beam_width // 2 + 1)
        
        for i in range(beam_width):
            temp = temperatures[i % len(temperatures)]
            tasks.append(model.generate(prompt, max_tokens=200, temperature=temp))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        candidates = []
        for result in results:
            if not isinstance(result, Exception) and result:
                candidates.append(result.strip())
        
        return candidates[:beam_width]
    
    async def rerank(self, query: str, candidates: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """Rerank candidates by relevance"""
        if not self.reranker or not candidates:
            # Fallback: return candidates with uniform scores
            return [(cand, 0.5) for cand in candidates]
        
        try:
            ranked_indices = await self.reranker.rerank(query, candidates, top_k)
            return [(candidates[idx], score) for idx, score in ranked_indices]
        except Exception:
            # Fallback on reranker failure
            return [(cand, 0.5) for cand in candidates]
    
    async def stitch(self, prompt: str, max_tokens: int = 1000) -> str:
        """Use stronger model for reasoning and stitching"""
        model = self.get_model(self.config.stitch_model)
        return await model.generate(prompt, max_tokens=max_tokens, temperature=0.3)


# Scoring functions for internal use
def calculate_expected_info_gain(question_text: str, context: str = "") -> float:
    """
    Calculate expected information gain for a question
    This is a simplified heuristic - in production, this could use learned models
    """
    # Simple heuristics based on question characteristics
    score = 0.0
    
    # Questions with quantitative elements tend to have higher info gain
    if any(word in question_text.lower() for word in ["how much", "how many", "what percentage", "rate", "cost"]):
        score += 0.3
    
    # Questions about comparisons/trade-offs
    if any(word in question_text.lower() for word in ["vs", "versus", "compare", "better", "trade-off"]):
        score += 0.2
    
    # Questions about specific processes/mechanisms
    if any(word in question_text.lower() for word in ["how does", "what happens", "process", "mechanism"]):
        score += 0.25
    
    # Questions about constraints/requirements
    if any(word in question_text.lower() for word in ["requirement", "constraint", "limit", "threshold"]):
        score += 0.2
    
    # Longer, more specific questions tend to have higher gain
    if len(question_text.split()) > 8:
        score += 0.1
    
    # Normalize to [0, 1]
    return min(score, 1.0)


def calculate_coherence(question: str, parent_questions: List[str], lane_thesis: str = "") -> float:
    """
    Calculate coherence of a question with its context
    Simple keyword overlap heuristic - could be enhanced with embeddings
    """
    if not parent_questions and not lane_thesis:
        return 0.5  # Neutral for root questions
    
    question_words = set(question.lower().split())
    
    # Check overlap with parent questions
    parent_overlap = 0.0
    if parent_questions:
        all_parent_words = set()
        for parent in parent_questions:
            all_parent_words.update(parent.lower().split())
        
        if all_parent_words:
            overlap = len(question_words.intersection(all_parent_words))
            parent_overlap = overlap / len(all_parent_words.union(question_words))
    
    # Check alignment with lane thesis
    thesis_overlap = 0.0
    if lane_thesis:
        thesis_words = set(lane_thesis.lower().split())
        if thesis_words:
            overlap = len(question_words.intersection(thesis_words))
            thesis_overlap = overlap / len(thesis_words.union(question_words))
    
    # Combine scores
    return (parent_overlap + thesis_overlap) / (2 if lane_thesis else 1)


def calculate_effort_penalty(question: str) -> float:
    """
    Estimate effort required to answer question
    Higher values = more effort = higher penalty
    """
    effort = 0.0
    
    # Questions requiring research/external data
    if any(word in question.lower() for word in ["research", "study", "data", "statistics", "market"]):
        effort += 0.3
    
    # Questions requiring complex analysis
    if any(word in question.lower() for word in ["analyze", "model", "simulate", "calculate", "optimize"]):
        effort += 0.2
    
    # Questions with multiple parts
    question_marks = question.count("?")
    if question_marks > 1:
        effort += 0.1 * (question_marks - 1)
    
    # Very long questions tend to be more complex
    if len(question.split()) > 15:
        effort += 0.1
    
    return min(effort, 0.8)  # Cap at 0.8 to avoid complete penalties