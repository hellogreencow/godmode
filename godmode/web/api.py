"""
API routes for the GodMode web application.

This module provides REST API endpoints for interacting with the reasoning engine
and hierarchical models.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from godmode.core.engine import GodModeEngine
from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel
from godmode.models.core import Problem, ReasoningType, CognitiveLevel
from godmode.models.commands import ReasoningCommand
from godmode.models.responses import ReasoningResponse


class ProblemRequest(BaseModel):
    """Request model for problem solving."""
    problem: str
    reasoning_type: ReasoningType = ReasoningType.HIERARCHICAL
    context: Optional[Dict[str, Any]] = None
    max_time: Optional[float] = None
    min_confidence: float = 0.7
    cognitive_levels: Optional[List[CognitiveLevel]] = None


class SolutionResponse(BaseModel):
    """Response model for solutions."""
    solutions: List[Dict[str, Any]]
    reasoning_trace: Dict[str, Any]
    confidence: float
    processing_time: float
    level_representations: Optional[Dict[str, Any]] = None


class StatisticsResponse(BaseModel):
    """Response model for system statistics."""
    engine_stats: Dict[str, Any]
    memory_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]


def create_api_router(
    engine: GodModeEngine,
    hierarchical_model: HierarchicalReasoningModel,
) -> APIRouter:
    """Create API router with all endpoints."""
    
    router = APIRouter()
    
    @router.post("/solve", response_model=SolutionResponse)
    async def solve_problem(request: ProblemRequest):
        """Solve a problem using the reasoning engine."""
        try:
            # Create problem object
            problem = Problem(
                title="API Problem",
                description=request.problem,
                problem_type="general",
                domain="general",
            )
            
            # Solve using main engine
            response = await engine.solve_problem(
                problem=problem,
                context=request.context,
                reasoning_type=request.reasoning_type,
                max_time=request.max_time,
                min_confidence=request.min_confidence,
            )
            
            # Convert to API response
            solutions = []
            if response.solution:
                solutions.append(response.solution.model_dump())
            
            for alt_solution in response.alternative_solutions:
                solutions.append(alt_solution.model_dump())
            
            return SolutionResponse(
                solutions=solutions,
                reasoning_trace=response.reasoning_trace.model_dump() if response.reasoning_trace else {},
                confidence=response.solution.confidence if response.solution else 0.0,
                processing_time=response.processing_time or 0.0,
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/solve/hierarchical", response_model=SolutionResponse)
    async def solve_with_hierarchical_model(request: ProblemRequest):
        """Solve a problem using the experimental hierarchical model."""
        try:
            # Use hierarchical model
            result = hierarchical_model.solve_problem(
                problem=request.problem,
                context=request.context,
            )
            
            # Convert solutions
            solutions = [solution.model_dump() for solution in result["solutions"]]
            
            return SolutionResponse(
                solutions=solutions,
                reasoning_trace=result["reasoning_trace"],
                confidence=result["confidence"],
                processing_time=0.0,  # Would be measured in full implementation
                level_representations=result.get("level_representations"),
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/statistics", response_model=StatisticsResponse)
    async def get_statistics():
        """Get system statistics and performance metrics."""
        try:
            engine_stats = engine.get_statistics()
            memory_stats = engine.memory.get_statistics()
            
            performance_metrics = {
                "active_sessions": len(engine._active_sessions),
                "total_requests": engine_stats.get("total_problems_solved", 0),
                "success_rate": engine_stats.get("success_rate", 0.0),
                "average_quality": engine_stats.get("average_quality", 0.0),
            }
            
            return StatisticsResponse(
                engine_stats=engine_stats,
                memory_stats=memory_stats,
                performance_metrics=performance_metrics,
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "engine_active": True,
            "model_loaded": hierarchical_model is not None,
        }
    
    @router.post("/memory/store")
    async def store_memory(
        content: Dict[str, Any],
        memory_type: str = "working",
        importance: float = 0.5,
    ):
        """Store content in memory system."""
        try:
            item_id = await engine.memory.store(
                content=content,
                memory_type=memory_type,
                importance=importance,
            )
            
            return {"item_id": str(item_id), "status": "stored"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/memory/retrieve")
    async def retrieve_memory(
        query: str,
        memory_types: Optional[List[str]] = None,
        top_k: int = 5,
    ):
        """Retrieve items from memory system."""
        try:
            results = await engine.memory.retrieve(
                query=query,
                memory_types=memory_types,
                top_k=top_k,
            )
            
            return {
                "results": [
                    {
                        "id": str(item.id),
                        "content": item.content,
                        "memory_type": item.memory_type,
                        "confidence": item.get_activation(),
                    }
                    for item in results
                ]
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models/info")
    async def get_model_info():
        """Get information about loaded models."""
        return {
            "hierarchical_model": {
                "embedding_dim": hierarchical_model.embedding_dim,
                "device": hierarchical_model.device,
                "use_pretrained": hierarchical_model.use_pretrained,
                "config": hierarchical_model.config,
            },
            "engine": {
                "memory_size": engine.memory_size,
                "reasoning_depth": engine.reasoning_depth,
                "max_parallel_branches": engine.max_parallel_branches,
                "enable_gpu": engine.enable_gpu,
                "device": str(engine.device),
            },
        }
    
    @router.post("/models/save")
    async def save_model(path: str):
        """Save the hierarchical model."""
        try:
            hierarchical_model.save_model(path)
            return {"status": "saved", "path": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/models/load")
    async def load_model(path: str):
        """Load a hierarchical model."""
        try:
            hierarchical_model.load_model(path)
            return {"status": "loaded", "path": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/reasoning/levels")
    async def get_reasoning_levels():
        """Get available cognitive levels."""
        return {
            "levels": [
                {
                    "name": level.value,
                    "description": _get_level_description(level),
                }
                for level in CognitiveLevel
            ]
        }
    
    @router.get("/reasoning/types")
    async def get_reasoning_types():
        """Get available reasoning types."""
        return {
            "types": [
                {
                    "name": rtype.value,
                    "description": _get_reasoning_type_description(rtype),
                }
                for rtype in ReasoningType
            ]
        }
    
    def _get_level_description(level: CognitiveLevel) -> str:
        """Get description for cognitive level."""
        descriptions = {
            CognitiveLevel.METACOGNITIVE: "Strategic planning and meta-reasoning",
            CognitiveLevel.EXECUTIVE: "Goal management and resource allocation", 
            CognitiveLevel.OPERATIONAL: "Task execution and procedure application",
            CognitiveLevel.REACTIVE: "Immediate responses and pattern matching",
        }
        return descriptions.get(level, "Unknown level")
    
    def _get_reasoning_type_description(rtype: ReasoningType) -> str:
        """Get description for reasoning type."""
        descriptions = {
            ReasoningType.FORWARD: "Forward chaining from facts to conclusions",
            ReasoningType.BACKWARD: "Backward chaining from goals to facts",
            ReasoningType.BIDIRECTIONAL: "Combined forward and backward reasoning",
            ReasoningType.HIERARCHICAL: "Multi-level hierarchical reasoning",
            ReasoningType.ABDUCTIVE: "Inference to the best explanation",
            ReasoningType.ANALOGICAL: "Reasoning by analogy and similarity",
            ReasoningType.CAUSAL: "Causal reasoning and inference",
            ReasoningType.TEMPORAL: "Temporal and sequential reasoning",
        }
        return descriptions.get(rtype, "Unknown reasoning type")
    
    return router