"""
Response models for the GodMode system.

This module defines response structures for various system operations,
providing structured and type-safe response handling.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from uuid import UUID

from pydantic import Field, computed_field

from godmode.models.core import (
    BaseGodModeModel,
    Solution,
    ReasoningTrace,
    CognitiveState,
    MemoryState,
)


class ResponseStatus(str, Enum):
    """Response status codes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ResponseType(str, Enum):
    """Types of responses."""
    REASONING = "reasoning"
    MEMORY = "memory"
    QUERY = "query"
    UPDATE = "update"
    ANALYSIS = "analysis"
    STREAMING = "streaming"
    BATCH = "batch"


class Response(BaseGodModeModel):
    """Base response class for all system operations."""
    
    response_type: ResponseType
    status: ResponseStatus
    
    # Request context
    request_id: UUID = Field(..., description="ID of the original request")
    command_id: Optional[UUID] = Field(None, description="ID of the executed command")
    
    # Timing information
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    started_at: Optional[datetime] = Field(None, description="When processing started")
    completed_at: Optional[datetime] = Field(None, description="When processing completed")
    
    # Messages and feedback
    message: Optional[str] = Field(None, description="Human-readable message")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @computed_field
    @property
    def is_successful(self) -> bool:
        """Check if the response indicates success."""
        return self.status in [ResponseStatus.SUCCESS, ResponseStatus.PARTIAL_SUCCESS]
    
    @computed_field
    @property
    def has_errors(self) -> bool:
        """Check if the response has errors."""
        return self.status in [ResponseStatus.FAILURE, ResponseStatus.ERROR]


class SuccessResponse(Response):
    """Response for successful operations."""
    
    status: ResponseStatus = ResponseStatus.SUCCESS
    
    # Result data
    data: Dict[str, Any] = Field(default_factory=dict, description="Response data")
    
    # Quality metrics
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in result")
    completeness: float = Field(1.0, ge=0.0, le=1.0, description="Result completeness")
    
    # Performance metrics
    cpu_time: Optional[float] = Field(None, ge=0.0, description="CPU time used")
    memory_peak: Optional[int] = Field(None, ge=0, description="Peak memory usage")
    
    def add_data(self, key: str, value: Any) -> None:
        """Add data to the response."""
        self.data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from the response."""
        return self.data.get(key, default)


class ErrorResponse(Response):
    """Response for failed operations."""
    
    status: ResponseStatus = ResponseStatus.ERROR
    
    # Error information
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Detailed error message")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    
    # Stack trace and debugging
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    debug_info: Dict[str, Any] = Field(default_factory=dict, description="Debug information")
    
    # Recovery suggestions
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested recovery actions")
    retry_after: Optional[float] = Field(None, ge=0.0, description="Retry delay in seconds")
    
    @classmethod
    def from_exception(
        cls, 
        request_id: UUID, 
        exception: Exception, 
        error_code: Optional[str] = None
    ) -> ErrorResponse:
        """Create error response from exception."""
        import traceback
        
        return cls(
            request_id=request_id,
            response_type=ResponseType.REASONING,  # Default type
            error_code=error_code or type(exception).__name__,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            debug_info={"exception_type": type(exception).__name__}
        )


class ReasoningResponse(SuccessResponse):
    """Response for reasoning operations."""
    
    response_type: ResponseType = ResponseType.REASONING
    
    # Core results
    solution: Optional[Solution] = Field(None, description="Generated solution")
    reasoning_trace: Optional[ReasoningTrace] = Field(None, description="Reasoning trace")
    final_cognitive_state: Optional[CognitiveState] = Field(None, description="Final cognitive state")
    
    # Alternative solutions
    alternative_solutions: List[Solution] = Field(
        default_factory=list, 
        description="Alternative solutions found"
    )
    
    # Reasoning quality metrics
    reasoning_coherence: float = Field(0.0, ge=0.0, le=1.0, description="Coherence of reasoning")
    reasoning_consistency: float = Field(0.0, ge=0.0, le=1.0, description="Consistency of reasoning")
    reasoning_efficiency: float = Field(0.0, ge=0.0, le=1.0, description="Efficiency of reasoning")
    
    # Process statistics
    total_steps: int = Field(0, ge=0, description="Total reasoning steps")
    successful_steps: int = Field(0, ge=0, description="Successful reasoning steps")
    backtrack_count: int = Field(0, ge=0, description="Number of backtracks")
    
    # Hierarchical information
    levels_used: List[str] = Field(default_factory=list, description="Cognitive levels used")
    cross_level_interactions: int = Field(0, ge=0, description="Cross-level interactions")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate reasoning success rate."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps
    
    @computed_field
    @property
    def overall_quality(self) -> float:
        """Calculate overall reasoning quality."""
        if not self.solution:
            return 0.0
        
        solution_quality = self.solution.get_overall_quality()
        process_quality = (
            self.reasoning_coherence + 
            self.reasoning_consistency + 
            self.reasoning_efficiency
        ) / 3.0
        
        return (solution_quality + process_quality) / 2.0


class MemoryResponse(SuccessResponse):
    """Response for memory operations."""
    
    response_type: ResponseType = ResponseType.MEMORY
    
    # Memory operation results
    retrieved_items: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Retrieved memory items"
    )
    stored_items: List[str] = Field(default_factory=list, description="IDs of stored items")
    updated_items: List[str] = Field(default_factory=list, description="IDs of updated items")
    deleted_items: List[str] = Field(default_factory=list, description="IDs of deleted items")
    
    # Memory state
    current_memory_state: Optional[MemoryState] = Field(None, description="Current memory state")
    memory_utilization: float = Field(0.0, ge=0.0, le=1.0, description="Memory utilization")
    
    # Search and retrieval metrics
    search_precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Search precision")
    search_recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Search recall")
    average_relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average relevance")
    
    def get_retrieval_count(self) -> int:
        """Get number of retrieved items."""
        return len(self.retrieved_items)
    
    def get_storage_count(self) -> int:
        """Get number of stored items."""
        return len(self.stored_items)


class QueryResponse(SuccessResponse):
    """Response for query operations."""
    
    response_type: ResponseType = ResponseType.QUERY
    
    # Query results
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    total_matches: int = Field(0, ge=0, description="Total number of matches")
    
    # Result metadata
    result_types: List[str] = Field(default_factory=list, description="Types of results returned")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence scores")
    relevance_scores: List[float] = Field(default_factory=list, description="Relevance scores")
    
    # Query analysis
    query_complexity: float = Field(0.0, ge=0.0, le=1.0, description="Query complexity score")
    interpretation_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Query interpretation confidence")
    
    # Performance metrics
    index_hits: int = Field(0, ge=0, description="Number of index hits")
    cache_hits: int = Field(0, ge=0, description="Number of cache hits")
    
    @computed_field
    @property
    def has_results(self) -> bool:
        """Check if query returned results."""
        return len(self.results) > 0
    
    @computed_field
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    @computed_field
    @property
    def average_relevance(self) -> float:
        """Calculate average relevance score."""
        if not self.relevance_scores:
            return 0.0
        return sum(self.relevance_scores) / len(self.relevance_scores)


class StreamingResponse(Response):
    """Response for streaming operations."""
    
    response_type: ResponseType = ResponseType.STREAMING
    status: ResponseStatus = ResponseStatus.SUCCESS
    
    # Streaming configuration
    stream_id: UUID = Field(..., description="Unique stream identifier")
    chunk_size: int = Field(1024, ge=1, description="Size of each chunk")
    total_chunks: Optional[int] = Field(None, ge=0, description="Total expected chunks")
    
    # Current chunk information
    current_chunk: int = Field(0, ge=0, description="Current chunk number")
    chunk_data: Dict[str, Any] = Field(default_factory=dict, description="Current chunk data")
    
    # Stream state
    is_complete: bool = Field(False, description="Whether stream is complete")
    buffer_size: int = Field(0, ge=0, description="Current buffer size")
    
    # Performance metrics
    throughput: Optional[float] = Field(None, ge=0.0, description="Data throughput")
    latency: Optional[float] = Field(None, ge=0.0, description="Current latency")
    
    @computed_field
    @property
    def progress(self) -> float:
        """Calculate stream progress as percentage."""
        if not self.total_chunks:
            return 0.0
        return min(1.0, self.current_chunk / self.total_chunks)
    
    def add_chunk(self, data: Dict[str, Any]) -> None:
        """Add data chunk to the stream."""
        self.current_chunk += 1
        self.chunk_data = data
        self.is_complete = (
            self.total_chunks is not None and 
            self.current_chunk >= self.total_chunks
        )


class BatchResponse(Response):
    """Response for batch operations."""
    
    response_type: ResponseType = ResponseType.BATCH
    
    # Batch results
    individual_responses: List[Response] = Field(
        default_factory=list, 
        description="Individual command responses"
    )
    
    # Batch statistics
    total_commands: int = Field(0, ge=0, description="Total number of commands")
    successful_commands: int = Field(0, ge=0, description="Number of successful commands")
    failed_commands: int = Field(0, ge=0, description="Number of failed commands")
    
    # Execution information
    execution_order: List[UUID] = Field(default_factory=list, description="Command execution order")
    parallel_executions: int = Field(0, ge=0, description="Number of parallel executions")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate batch success rate."""
        if self.total_commands == 0:
            return 0.0
        return self.successful_commands / self.total_commands
    
    @computed_field
    @property
    def overall_status(self) -> ResponseStatus:
        """Determine overall batch status."""
        if self.failed_commands == 0:
            return ResponseStatus.SUCCESS
        elif self.successful_commands > 0:
            return ResponseStatus.PARTIAL_SUCCESS
        else:
            return ResponseStatus.FAILURE
    
    def add_response(self, response: Response) -> None:
        """Add individual response to batch."""
        self.individual_responses.append(response)
        self.total_commands += 1
        
        if response.is_successful:
            self.successful_commands += 1
        else:
            self.failed_commands += 1
            
        # Update overall status
        self.status = self.overall_status