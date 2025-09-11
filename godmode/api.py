"""
FastAPI interface for GODMODE
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging

from .engine import GodmodeEngine
from .schemas import Command, GodmodeResponse, InitCommand, AdvanceCommand, ContinueCommand, SummarizeCommand, RegraftCommand, MergeCommand
from .models import ModelConfig
from .ladder_generator import LadderConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GODMODE",
    description="A superhuman, ontological Question Foresight Engine",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[GodmodeEngine] = None


async def get_engine() -> GodmodeEngine:
    """Get or create the GODMODE engine instance"""
    global engine
    if engine is None:
        model_config = ModelConfig()
        ladder_config = LadderConfig()
        engine = GodmodeEngine(model_config, ladder_config)
    return engine


# Request/Response models for API
class InitRequest(BaseModel):
    current_question: str
    context: Optional[str] = None
    budgets: Optional[Dict[str, Any]] = None


class AdvanceRequest(BaseModel):
    node_id: str
    user_answer: Optional[str] = None


class ContinueRequest(BaseModel):
    thread_id: str


class SummarizeRequest(BaseModel):
    thread_id: str


class RegraftRequest(BaseModel):
    from_node_id: str
    to_lane_id: str


class MergeRequest(BaseModel):
    thread_ids: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    engine_ready: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        engine_instance = await get_engine()
        engine_ready = engine_instance is not None
    except Exception:
        engine_ready = False
    
    return HealthResponse(
        status="healthy" if engine_ready else "degraded",
        version="0.1.0",
        engine_ready=engine_ready
    )


@app.post("/init", response_model=GodmodeResponse)
async def init_exploration(request: InitRequest) -> GodmodeResponse:
    """Initialize new question exploration"""
    try:
        engine_instance = await get_engine()
        
        command = Command(
            command_type="INIT",
            data=InitCommand(
                current_question=request.current_question,
                context=request.context,
                budgets=request.budgets
            )
        )
        
        response = await engine_instance.process_command(command)
        logger.info(f"Initialized exploration for: {request.current_question[:50]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in init_exploration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/advance", response_model=GodmodeResponse)
async def advance_exploration(request: AdvanceRequest) -> GodmodeResponse:
    """Advance exploration around a chosen node"""
    try:
        engine_instance = await get_engine()
        
        command = Command(
            command_type="ADVANCE",
            data=AdvanceCommand(
                node_id=request.node_id,
                user_answer=request.user_answer
            )
        )
        
        response = await engine_instance.process_command(command)
        logger.info(f"Advanced exploration at node: {request.node_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in advance_exploration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/continue", response_model=GodmodeResponse)
async def continue_thread(request: ContinueRequest) -> GodmodeResponse:
    """Continue exploration along the deepest promising thread"""
    try:
        engine_instance = await get_engine()
        
        command = Command(
            command_type="CONTINUE",
            data=ContinueCommand(thread_id=request.thread_id)
        )
        
        response = await engine_instance.process_command(command)
        logger.info(f"Continued thread: {request.thread_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in continue_thread: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=GodmodeResponse)
async def summarize_thread(request: SummarizeRequest) -> GodmodeResponse:
    """Summarize a thread path and recommend next step"""
    try:
        engine_instance = await get_engine()
        
        command = Command(
            command_type="SUMMARIZE",
            data=SummarizeCommand(thread_id=request.thread_id)
        )
        
        response = await engine_instance.process_command(command)
        logger.info(f"Summarized thread: {request.thread_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in summarize_thread: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regraft", response_model=GodmodeResponse)
async def regraft_branch(request: RegraftRequest) -> GodmodeResponse:
    """Move a sub-branch to a different lane for better coherence"""
    try:
        engine_instance = await get_engine()
        
        command = Command(
            command_type="REGRAFT",
            data=RegraftCommand(
                from_node_id=request.from_node_id,
                to_lane_id=request.to_lane_id
            )
        )
        
        response = await engine_instance.process_command(command)
        logger.info(f"Regrafted {request.from_node_id} to {request.to_lane_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in regraft_branch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/merge", response_model=GodmodeResponse)
async def merge_threads(request: MergeRequest) -> GodmodeResponse:
    """Merge concurrent branches into a unified path"""
    try:
        engine_instance = await get_engine()
        
        command = Command(
            command_type="MERGE",
            data=MergeCommand(thread_ids=request.thread_ids)
        )
        
        response = await engine_instance.process_command(command)
        logger.info(f"Merged threads: {request.thread_ids}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in merge_threads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "GODMODE",
        "tagline": "See the questions before you ask them.",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)