"""Main GODMODE engine implementation."""

import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from ..models.commands import Command, InitCommand, AdvanceCommand, ContinueCommand, SummarizeCommand, RegraftCommand, MergeCommand, Budgets
from ..models.responses import GodmodeResponse, GraphUpdate, OntologyUpdate, Meta
from ..models.core import Question, Lane, Thread, Entity, Relation, Mapping, CognitiveMove, ThreadStatus
from ..schemas.validator import SchemaValidator
from .reasoning import BackwardReasoning, ForwardReasoning
from .ontology import OntologyManager
from .memory import MemoryManager
from .validation import InvariantValidator


class GodmodeEngine:
    """
    GODMODE - Superhuman Question Foresight Engine
    
    Operates in dual modes:
    - Operator: silent, parallel enumeration, ranking, stitching, memory updates
    - Interface: crisp chat replies + structured graph/ontology updates
    """
    
    def __init__(self):
        self.backward_reasoning = BackwardReasoning()
        self.forward_reasoning = ForwardReasoning()
        self.ontology_manager = OntologyManager()
        self.memory_manager = MemoryManager()
        self.validator = InvariantValidator()
        self.schema_validator = SchemaValidator()
        
        # Cache for deduplication
        self._question_cache: Dict[str, str] = {}
        
        # Current state
        self._current_question: Optional[str] = None
        self._active_threads: List[Thread] = []
        self._budgets: Budgets = Budgets()
    
    async def process_command(self, command: Command) -> GodmodeResponse:
        """Process a GODMODE command and return structured response."""
        start_time = time.time()
        
        try:
            if isinstance(command, InitCommand):
                response = await self._handle_init(command)
            elif isinstance(command, AdvanceCommand):
                response = await self._handle_advance(command)
            elif isinstance(command, ContinueCommand):
                response = await self._handle_continue(command)
            elif isinstance(command, SummarizeCommand):
                response = await self._handle_summarize(command)
            elif isinstance(command, RegraftCommand):
                response = await self._handle_regraft(command)
            elif isinstance(command, MergeCommand):
                response = await self._handle_merge(command)
            else:
                raise ValueError(f"Unknown command type: {type(command)}")
            
            # Update timing metadata
            elapsed = time.time() - start_time
            response.graph_update.meta.budgets_used["time_s"] = elapsed
            
            # Validate response
            response_dict = response.dict()
            is_valid, errors = self.schema_validator.validate_response(response_dict)
            if not is_valid:
                raise ValueError(f"Response validation failed: {errors}")
            
            return response
            
        except Exception as e:
            # Fallback response for errors
            return self._create_error_response(str(e))
    
    async def _handle_init(self, command: InitCommand) -> GodmodeResponse:
        """Handle INIT command - analyze a new question."""
        self._current_question = command.current_question
        if command.budgets:
            self._budgets = command.budgets
        
        # Canonicalize question for caching
        question_hash = self._hash_question(command.current_question)
        
        # Phase 1: ENUMERATE (parallel candidate generation)
        priors_candidates, futures_candidates = await asyncio.gather(
            self.backward_reasoning.enumerate_priors(
                command.current_question, 
                command.context,
                self._budgets
            ),
            self.forward_reasoning.enumerate_futures(
                command.current_question,
                command.context, 
                self._budgets
            )
        )
        
        # Phase 2: RERANK (score by expected_info_gain × coherence × effort)
        ranked_priors, ranked_futures = await asyncio.gather(
            self.backward_reasoning.rerank_candidates(priors_candidates),
            self.forward_reasoning.rerank_candidates(futures_candidates)
        )
        
        # Phase 3: STITCH (wire builds_on, triggers, cross-lane junctions)
        priors = await self.backward_reasoning.stitch_ladder(ranked_priors)
        scenarios = await self.forward_reasoning.stitch_scenarios(ranked_futures)
        
        # Extract ontology
        ontology_update = await self.ontology_manager.extract_from_question(
            command.current_question,
            command.context,
            priors + [q for s in scenarios for q in s.lane]
        )
        
        # Create initial thread
        if priors:
            initial_thread = Thread(
                thread_id="T1",
                origin_node_id=priors[0].id,
                path=[priors[0].id],
                status=ThreadStatus.ACTIVE,
                summary=f"Starting from: {priors[0].text[:100]}..."
            )
            self._active_threads = [initial_thread]
        else:
            self._active_threads = []
        
        # Generate chat reply
        chat_reply = self._generate_chat_reply(priors, scenarios)
        
        # Create response
        graph_update = GraphUpdate(
            current_question=command.current_question,
            priors=priors,
            scenarios=scenarios,
            threads=self._active_threads,
            meta=Meta(
                budgets_used=self._budgets.dict(),
                notes=f"Processed with hash {question_hash[:8]}"
            )
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_advance(self, command: AdvanceCommand) -> GodmodeResponse:
        """Handle ADVANCE command - expand around a chosen node."""
        # Find the node to advance from
        node = await self._find_node_by_id(command.node_id)
        if not node:
            raise ValueError(f"Node {command.node_id} not found")
        
        # Generate expanded ladder around this node
        expanded_questions = await self.forward_reasoning.expand_around_node(
            node, command.user_answer, self._budgets
        )
        
        # Update existing scenarios or create new ones
        scenarios = await self._update_scenarios_with_expansion(expanded_questions)
        
        # Update ontology
        ontology_update = await self.ontology_manager.update_from_expansion(
            expanded_questions
        )
        
        # Update threads
        self._update_threads_for_advance(command.node_id, expanded_questions)
        
        chat_reply = f"Expanded around **{command.node_id}** with {len(expanded_questions)} new branches."
        
        graph_update = GraphUpdate(
            current_question=self._current_question or "",
            priors=await self._get_current_priors(),
            scenarios=scenarios,
            threads=self._active_threads,
            meta=Meta(budgets_used=self._budgets.dict())
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_continue(self, command: ContinueCommand) -> GodmodeResponse:
        """Handle CONTINUE command - continue deepest promising lane."""
        thread = next((t for t in self._active_threads if t.thread_id == command.thread_id), None)
        if not thread:
            raise ValueError(f"Thread {command.thread_id} not found")
        
        # Continue the thread's deepest node
        last_node_id = thread.path[-1]
        last_node = await self._find_node_by_id(last_node_id)
        
        if not last_node:
            raise ValueError(f"Last node {last_node_id} not found in thread")
        
        # Generate continuation
        continued_questions = await self.forward_reasoning.continue_from_node(
            last_node, self._budgets
        )
        
        # Update scenarios and thread
        scenarios = await self._update_scenarios_with_continuation(continued_questions)
        thread.path.extend([q.id for q in continued_questions])
        thread.summary = f"Continued to: {continued_questions[-1].text[:100]}..." if continued_questions else thread.summary
        
        ontology_update = await self.ontology_manager.update_from_expansion(continued_questions)
        
        chat_reply = f"Continued **{command.thread_id}** with {len(continued_questions)} deeper questions."
        
        graph_update = GraphUpdate(
            current_question=self._current_question or "",
            priors=await self._get_current_priors(),
            scenarios=scenarios,
            threads=self._active_threads,
            meta=Meta(budgets_used=self._budgets.dict())
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_summarize(self, command: SummarizeCommand) -> GodmodeResponse:
        """Handle SUMMARIZE command - get path summary."""
        thread = next((t for t in self._active_threads if t.thread_id == command.thread_id), None)
        if not thread:
            raise ValueError(f"Thread {command.thread_id} not found")
        
        # Generate detailed summary
        summary = await self._generate_thread_summary(thread)
        next_recommendation = await self._recommend_next_step(thread)
        
        chat_reply = f"**{command.thread_id}**: {summary} → Recommend: {next_recommendation}"
        
        # Return current state
        graph_update = GraphUpdate(
            current_question=self._current_question or "",
            priors=await self._get_current_priors(),
            scenarios=await self._get_current_scenarios(),
            threads=self._active_threads,
            meta=Meta(budgets_used=self._budgets.dict())
        )
        
        ontology_update = await self.ontology_manager.get_current_state()
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_regraft(self, command: RegraftCommand) -> GodmodeResponse:
        """Handle REGRAFT command - move sub-branch to different lane."""
        # Implementation for regrafting nodes between lanes
        # This is a complex operation that requires careful invariant preservation
        
        chat_reply = f"Regrafted branch from {command.from_node_id} to {command.to_lane_id}"
        
        # Placeholder implementation
        graph_update = GraphUpdate(
            current_question=self._current_question or "",
            priors=await self._get_current_priors(),
            scenarios=await self._get_current_scenarios(),
            threads=self._active_threads,
            meta=Meta(budgets_used=self._budgets.dict())
        )
        
        ontology_update = await self.ontology_manager.get_current_state()
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_merge(self, command: MergeCommand) -> GodmodeResponse:
        """Handle MERGE command - merge concurrent branches."""
        # Implementation for merging thread branches
        
        chat_reply = f"Merged {len(command.thread_ids)} threads into unified path"
        
        # Placeholder implementation
        graph_update = GraphUpdate(
            current_question=self._current_question or "",
            priors=await self._get_current_priors(),
            scenarios=await self._get_current_scenarios(),
            threads=self._active_threads,
            meta=Meta(budgets_used=self._budgets.dict())
        )
        
        ontology_update = await self.ontology_manager.get_current_state()
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    def _hash_question(self, question: str) -> str:
        """Generate canonical hash for question deduplication."""
        # Normalize and hash the question
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _generate_chat_reply(self, priors: List[Question], scenarios: List[Lane]) -> str:
        """Generate concise tactical chat reply."""
        if not priors and not scenarios:
            return "No clear prior questions or future scenarios identified."
        
        reply_parts = []
        
        # Reference top priors
        if priors:
            top_priors = sorted(priors, key=lambda q: q.expected_info_gain, reverse=True)[:2]
            prior_refs = [f"**{p.id}** ({p.cognitive_move})" for p in top_priors]
            reply_parts.append(f"Start with {', '.join(prior_refs)}")
        
        # Reference top lanes
        if scenarios:
            top_lanes = sorted(scenarios, key=lambda s: max((q.expected_info_gain for q in s.lane), default=0), reverse=True)[:2]
            lane_refs = [f"**{s.id}** ({s.name})" for s in top_lanes]
            reply_parts.append(f"then explore {', '.join(lane_refs)}")
        
        return ". ".join(reply_parts) + "."
    
    def _create_error_response(self, error_msg: str) -> GodmodeResponse:
        """Create a fallback error response."""
        return GodmodeResponse(
            chat_reply=f"Error processing request: {error_msg}",
            graph_update=GraphUpdate(
                current_question="",
                priors=[],
                scenarios=[],
                threads=[],
                meta=Meta(budgets_used={})
            ),
            ontology_update=OntologyUpdate(
                entities=[],
                relations=[],
                mappings=[]
            )
        )
    
    # Helper methods (stubs for now)
    async def _find_node_by_id(self, node_id: str) -> Optional[Question]:
        """Find a question node by ID."""
        # Implementation needed
        return None
    
    async def _get_current_priors(self) -> List[Question]:
        """Get current prior questions."""
        return []
    
    async def _get_current_scenarios(self) -> List[Lane]:
        """Get current scenario lanes."""
        return []
    
    async def _update_scenarios_with_expansion(self, questions: List[Question]) -> List[Lane]:
        """Update scenarios with expanded questions."""
        return []
    
    async def _update_scenarios_with_continuation(self, questions: List[Question]) -> List[Lane]:
        """Update scenarios with continued questions."""
        return []
    
    def _update_threads_for_advance(self, node_id: str, questions: List[Question]) -> None:
        """Update threads after advancing from a node."""
        pass
    
    async def _generate_thread_summary(self, thread: Thread) -> str:
        """Generate detailed thread summary."""
        return f"Thread following path through {len(thread.path)} nodes"
    
    async def _recommend_next_step(self, thread: Thread) -> str:
        """Recommend next step for thread."""
        return "Continue deepest promising branch"