"""Mock AI models and services for testing GODMODE."""

import random
import asyncio
from typing import List, Dict, Any, Optional
from ..models.core import Question, CognitiveMove


class MockEnumerationModel:
    """Mock model for Phase 1: ENUMERATE - candidate generation."""
    
    def __init__(self, response_delay: float = 0.1):
        self.response_delay = response_delay
        self.call_count = 0
        
    async def generate_candidates(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        num_candidates: int = 4
    ) -> List[str]:
        """Generate mock question candidates."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        # Generate diverse mock candidates based on prompt keywords
        templates = [
            "What exactly does '{topic}' mean in this context?",
            "How should we define the boundaries for '{topic}'?",
            "What metrics would help us measure '{topic}'?",
            "How does '{topic}' compare to alternative approaches?",
            "What scenarios should we consider for '{topic}'?",
            "What decision criteria should guide our '{topic}' choice?",
            "What are the next steps to implement '{topic}'?"
        ]
        
        # Extract a key term from the prompt
        words = prompt.split()
        topic = next((word for word in words if len(word) > 4), "approach")
        
        candidates = []
        for i in range(num_candidates):
            template = random.choice(templates)
            candidate = template.format(topic=topic)
            candidates.append(candidate)
        
        return candidates


class MockRerankingModel:
    """Mock model for Phase 2: RERANK - scoring candidates."""
    
    def __init__(self, response_delay: float = 0.05):
        self.response_delay = response_delay
        self.call_count = 0
    
    async def rank_candidates(
        self, 
        candidates: List[str], 
        query: str,
        features: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Rank candidates with mock scores."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        ranked_candidates = []
        
        for i, candidate in enumerate(candidates):
            # Mock scoring based on simple heuristics
            score = self._calculate_mock_score(candidate, query)
            
            ranked_candidates.append({
                "text": candidate,
                "score": score,
                "features": {
                    "length": len(candidate.split()),
                    "has_question_word": any(word in candidate.lower() 
                                           for word in ["what", "how", "why", "when", "where"]),
                    "specificity": candidate.count("specific") + candidate.count("exactly")
                }
            })
        
        # Sort by score descending
        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return ranked_candidates
    
    def _calculate_mock_score(self, candidate: str, query: str) -> float:
        """Calculate mock relevance score."""
        base_score = 0.5
        
        # Boost score for question words
        if any(word in candidate.lower() for word in ["what", "how", "why"]):
            base_score += 0.2
        
        # Boost score for specificity indicators
        if any(word in candidate.lower() for word in ["exactly", "specific", "measure"]):
            base_score += 0.15
        
        # Boost score for shared words with query
        query_words = set(query.lower().split())
        candidate_words = set(candidate.lower().split())
        overlap = len(query_words & candidate_words)
        base_score += min(0.3, overlap * 0.05)
        
        # Add some randomness
        base_score += random.uniform(-0.1, 0.1)
        
        return min(1.0, max(0.0, base_score))


class MockStitchingModel:
    """Mock model for Phase 3: STITCH - building relationships."""
    
    def __init__(self, response_delay: float = 0.2):
        self.response_delay = response_delay
        self.call_count = 0
    
    async def build_relationships(
        self, 
        questions: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build mock relationships between questions."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        # Convert to Question objects with relationships
        structured_questions = []
        
        for i, q_data in enumerate(questions):
            # Determine cognitive move based on question content
            cognitive_move = self._infer_cognitive_move(q_data["text"])
            
            # Determine level based on cognitive move
            level = self._get_level_for_move(cognitive_move)
            
            # Build dependencies (builds_on)
            builds_on = []
            if i > 0 and level > 1:
                # Connect to previous questions at lower levels
                for j in range(i):
                    prev_q = structured_questions[j]
                    if prev_q["level"] < level:
                        builds_on.append(prev_q["id"])
                        break  # Connect to most recent lower level
            
            question_data = {
                "id": f"Q{i+1}",
                "text": q_data["text"],
                "level": level,
                "cognitive_move": cognitive_move,
                "builds_on": builds_on,
                "delta_nuance": f"Adds {cognitive_move.value} perspective",
                "expected_info_gain": q_data.get("score", 0.5),
                "confidence": min(1.0, q_data.get("score", 0.5) + 0.2),
                "triggers": [],
                "natural_end": False,
                "tags": [cognitive_move.value, "mock"]
            }
            
            structured_questions.append(question_data)
        
        return structured_questions
    
    def _infer_cognitive_move(self, text: str) -> CognitiveMove:
        """Infer cognitive move from question text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["what", "define", "mean"]):
            return CognitiveMove.DEFINE
        elif any(word in text_lower for word in ["boundaries", "scope", "within"]):
            return CognitiveMove.SCOPE
        elif any(word in text_lower for word in ["measure", "metrics", "quantify"]):
            return CognitiveMove.QUANTIFY
        elif any(word in text_lower for word in ["compare", "versus", "alternative"]):
            return CognitiveMove.COMPARE
        elif any(word in text_lower for word in ["scenario", "simulate", "model"]):
            return CognitiveMove.SIMULATE
        elif any(word in text_lower for word in ["decide", "choose", "should"]):
            return CognitiveMove.DECIDE
        elif any(word in text_lower for word in ["implement", "steps", "action"]):
            return CognitiveMove.COMMIT
        else:
            return CognitiveMove.DEFINE  # Default
    
    def _get_level_for_move(self, move: CognitiveMove) -> int:
        """Get typical level for a cognitive move."""
        move_levels = {
            CognitiveMove.DEFINE: 1,
            CognitiveMove.SCOPE: 2,
            CognitiveMove.QUANTIFY: 3,
            CognitiveMove.COMPARE: 4,
            CognitiveMove.SIMULATE: 5,
            CognitiveMove.DECIDE: 6,
            CognitiveMove.COMMIT: 7
        }
        return move_levels.get(move, 1)


class MockFactChecker:
    """Mock fact-checking service."""
    
    def __init__(self, response_delay: float = 0.3):
        self.response_delay = response_delay
        self.call_count = 0
    
    async def verify_claim(self, claim: str) -> Dict[str, Any]:
        """Mock fact verification."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        # Mock verification result
        confidence = random.uniform(0.3, 0.9)
        
        return {
            "claim": claim,
            "verified": confidence > 0.6,
            "confidence": confidence,
            "sources": [
                {"url": "https://example.com/source1", "title": "Mock Source 1"},
                {"url": "https://example.com/source2", "title": "Mock Source 2"}
            ] if confidence > 0.7 else [],
            "explanation": f"Mock verification with {confidence:.2f} confidence"
        }


class MockModelOrchestrator:
    """Orchestrator for mock AI models."""
    
    def __init__(self):
        self.enumeration_model = MockEnumerationModel()
        self.reranking_model = MockRerankingModel()
        self.stitching_model = MockStitchingModel()
        self.fact_checker = MockFactChecker()
        
        # Performance tracking
        self.total_calls = 0
        self.total_latency = 0.0
    
    async def enumerate_and_rank(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        num_candidates: int = 4
    ) -> List[Dict[str, Any]]:
        """Full enumerate -> rank pipeline."""
        start_time = asyncio.get_event_loop().time()
        
        # Phase 1: Enumerate
        candidates = await self.enumeration_model.generate_candidates(
            prompt, context, num_candidates
        )
        
        # Phase 2: Rerank
        ranked_candidates = await self.reranking_model.rank_candidates(
            candidates, prompt
        )
        
        # Phase 3: Stitch
        structured_questions = await self.stitching_model.build_relationships(
            ranked_candidates, context
        )
        
        # Track performance
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time
        self.total_calls += 1
        self.total_latency += latency
        
        return structured_questions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_calls": self.total_calls,
            "total_latency": self.total_latency,
            "avg_latency": self.total_latency / max(self.total_calls, 1),
            "model_calls": {
                "enumeration": self.enumeration_model.call_count,
                "reranking": self.reranking_model.call_count,
                "stitching": self.stitching_model.call_count,
                "fact_checking": self.fact_checker.call_count
            }
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_calls = 0
        self.total_latency = 0.0
        self.enumeration_model.call_count = 0
        self.reranking_model.call_count = 0
        self.stitching_model.call_count = 0
        self.fact_checker.call_count = 0