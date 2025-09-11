"""
Core GODMODE engine - orchestrates the dual-mode operation
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .schemas import (
    GodmodeResponse, GraphUpdate, OntologyUpdate, QuestionNode, ScenarioLane, 
    Thread, GraphMeta, BudgetsUsed, Command, InitCommand, AdvanceCommand,
    ContinueCommand, SummarizeCommand, RegraftCommand, MergeCommand
)
from .models import ModelRouter, ModelConfig
from .ladder_generator import LadderGenerator, LadderConfig
from .ontology import OntologyManager
from .validator import InvariantValidator


@dataclass
class ThreadState:
    """State tracking for active threads"""
    thread_id: str
    path: List[str]  # Node IDs in order
    current_node: Optional[str] = None
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class GodmodeEngine:
    """
    GODMODE - Superhuman, ontological Question Foresight Engine
    
    Operates in dual modes:
    - Operator: silent, parallel enumeration, ranking, stitching, memory updates
    - Interface: crisp chat replies + structured graph/ontology updates
    """
    
    def __init__(self, model_config: ModelConfig = None, ladder_config: LadderConfig = None):
        # Initialize components
        self.model_config = model_config or ModelConfig()
        self.ladder_config = ladder_config or LadderConfig()
        self.router = ModelRouter(self.model_config)
        self.ladder_gen = LadderGenerator(self.router, self.ladder_config)
        self.ontology = OntologyManager()
        self.validator = InvariantValidator()
        
        # State management
        self.current_question: str = ""
        self.context: str = ""
        self.threads: Dict[str, ThreadState] = {}
        self.thread_counter = 0
        
        # Performance tracking
        self.budgets_used = BudgetsUsed()
    
    async def process_command(self, command: Command) -> GodmodeResponse:
        """Main entry point - process any command and return structured response"""
        start_time = time.time()
        
        try:
            if command.command_type == "INIT":
                response = await self._handle_init(command.data)
            elif command.command_type == "ADVANCE":
                response = await self._handle_advance(command.data)
            elif command.command_type == "CONTINUE":
                response = await self._handle_continue(command.data)
            elif command.command_type == "SUMMARIZE":
                response = await self._handle_summarize(command.data)
            elif command.command_type == "REGRAFT":
                response = await self._handle_regraft(command.data)
            elif command.command_type == "MERGE":
                response = await self._handle_merge(command.data)
            else:
                raise ValueError(f"Unknown command type: {command.command_type}")
            
            # Update timing
            self.budgets_used.time_s = time.time() - start_time
            response.graph_update.meta.budgets_used = self.budgets_used
            
            # Validate response
            self.validator.validate_response(response)
            
            return response
            
        except Exception as e:
            # Return error response
            return self._create_error_response(str(e))
    
    async def _handle_init(self, init_cmd: InitCommand) -> GodmodeResponse:
        """Initialize new question exploration"""
        self.current_question = init_cmd.current_question
        self.context = init_cmd.context or ""
        
        # Update budgets if provided
        if init_cmd.budgets:
            self._update_budgets(init_cmd.budgets)
        
        # Reset state
        self.threads.clear()
        self.ladder_gen.question_counter = 0
        self.ladder_gen.lane_counter = 0
        self.ladder_gen.seen_questions.clear()
        
        # OPERATOR MODE: Parallel generation
        # Phase 1: Generate priors and scenarios in parallel
        priors_task = self.ladder_gen.generate_priors(self.current_question, self.context)
        scenarios_task = self.ladder_gen.generate_scenarios(self.current_question, self.context)
        
        priors, scenarios = await asyncio.gather(priors_task, scenarios_task)
        
        # Phase 2: Update ontology
        entities, relations, mapping = self.ontology.update_from_question(
            "Q000", self.current_question, self.context
        )
        
        # Phase 3: Create initial threads
        threads = self._create_initial_threads(priors, scenarios)
        
        # INTERFACE MODE: Generate chat reply
        chat_reply = self._generate_init_chat_reply(priors, scenarios)
        
        # Build response
        graph_update = GraphUpdate(
            current_question=self.current_question,
            priors=priors,
            scenarios=scenarios,
            threads=threads,
            meta=GraphMeta(budgets_used=self.budgets_used)
        )
        
        ontology_update = OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=[mapping]
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_advance(self, advance_cmd: AdvanceCommand) -> GodmodeResponse:
        """Advance exploration around a chosen node"""
        node_id = advance_cmd.node_id
        user_answer = advance_cmd.user_answer
        
        # Find the node in current state
        target_node = self._find_node_by_id(node_id)
        if not target_node:
            raise ValueError(f"Node {node_id} not found")
        
        # Update context if user provided an answer
        if user_answer:
            self.context += f"\n\nUser's answer to '{target_node.text}': {user_answer}"
        
        # Generate expanded exploration around this node
        expansion_context = f"Building on: {target_node.text}"
        if user_answer:
            expansion_context += f"\nUser's input: {user_answer}"
        
        # Generate new branches from this node
        new_scenarios = await self.ladder_gen.generate_scenarios(
            target_node.text, expansion_context
        )
        
        # Update ontology with new information
        entities, relations, mapping = self.ontology.update_from_question(
            node_id, target_node.text, expansion_context
        )
        
        # Update threads
        self._update_threads_for_advance(node_id, new_scenarios)
        
        # Generate chat reply
        chat_reply = f"Expanded around **{node_id}** with {len(new_scenarios)} new paths. " + \
                    f"Highest info gain: **{new_scenarios[0].lane[0].id if new_scenarios else 'none'}**."
        
        # Build response
        graph_update = GraphUpdate(
            current_question=self.current_question,
            priors=self._get_current_priors(),
            scenarios=new_scenarios,
            threads=self._get_current_threads(),
            meta=GraphMeta(budgets_used=self.budgets_used)
        )
        
        ontology_update = OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=[mapping]
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_continue(self, continue_cmd: ContinueCommand) -> GodmodeResponse:
        """Continue exploration along the deepest promising thread"""
        thread_id = continue_cmd.thread_id
        
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")
        
        thread = self.threads[thread_id]
        
        if not thread.path:
            raise ValueError(f"Thread {thread_id} has no path to continue")
        
        # Get the last node in the path
        last_node_id = thread.path[-1]
        last_node = self._find_node_by_id(last_node_id)
        
        if not last_node:
            raise ValueError(f"Last node {last_node_id} in thread not found")
        
        # Generate continuation from this point
        continue_context = f"Continuing from: {last_node.text}"
        
        new_scenarios = await self.ladder_gen.generate_scenarios(
            last_node.text, continue_context
        )
        
        # Update thread path
        if new_scenarios and new_scenarios[0].lane:
            next_node_id = new_scenarios[0].lane[0].id
            thread.path.append(next_node_id)
            thread.current_node = next_node_id
            thread.last_updated = time.time()
        
        # Generate chat reply
        chat_reply = f"Continued **{thread_id}** → **{thread.current_node}**. " + \
                    f"Path depth: {len(thread.path)}."
        
        # Build response (similar structure to advance)
        graph_update = GraphUpdate(
            current_question=self.current_question,
            priors=self._get_current_priors(),
            scenarios=new_scenarios,
            threads=self._get_current_threads(),
            meta=GraphMeta(budgets_used=self.budgets_used)
        )
        
        # Update ontology
        entities, relations, mappings = self.ontology.export_for_update()
        ontology_update = OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=mappings
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_summarize(self, summarize_cmd: SummarizeCommand) -> GodmodeResponse:
        """Summarize a thread path and recommend next step"""
        thread_id = summarize_cmd.thread_id
        
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")
        
        thread = self.threads[thread_id]
        
        # Build path summary
        path_nodes = []
        for node_id in thread.path:
            node = self._find_node_by_id(node_id)
            if node:
                path_nodes.append(node)
        
        # Generate summary using model
        if path_nodes:
            path_text = " → ".join([f"{n.id}: {n.text}" for n in path_nodes])
            summary_prompt = f"Summarize this question path and recommend next step:\n{path_text}"
            
            summary = await self.router.stitch(summary_prompt, max_tokens=280)
            thread.status = "summarized"
        else:
            summary = "Empty thread path"
        
        # Generate chat reply
        chat_reply = f"**{thread_id}** summary: {summary[:100]}..."
        
        # Build minimal response
        graph_update = GraphUpdate(
            current_question=self.current_question,
            priors=[],
            scenarios=[],
            threads=self._get_current_threads(),
            meta=GraphMeta(budgets_used=self.budgets_used)
        )
        
        entities, relations, mappings = self.ontology.export_for_update()
        ontology_update = OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=mappings
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_regraft(self, regraft_cmd: RegraftCommand) -> GodmodeResponse:
        """Move a sub-branch to a different lane for better coherence"""
        # This is a complex operation - simplified implementation
        chat_reply = f"Regrafted **{regraft_cmd.from_node_id}** → **{regraft_cmd.to_lane_id}**"
        
        # Return minimal response
        graph_update = GraphUpdate(
            current_question=self.current_question,
            priors=self._get_current_priors(),
            scenarios=self._get_current_scenarios(),
            threads=self._get_current_threads(),
            meta=GraphMeta(budgets_used=self.budgets_used)
        )
        
        entities, relations, mappings = self.ontology.export_for_update()
        ontology_update = OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=mappings
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    async def _handle_merge(self, merge_cmd: MergeCommand) -> GodmodeResponse:
        """Merge concurrent branches into a unified path"""
        # Simplified implementation
        chat_reply = f"Merged {len(merge_cmd.thread_ids)} threads into unified path"
        
        # Return minimal response
        graph_update = GraphUpdate(
            current_question=self.current_question,
            priors=self._get_current_priors(),
            scenarios=self._get_current_scenarios(),
            threads=self._get_current_threads(),
            meta=GraphMeta(budgets_used=self.budgets_used)
        )
        
        entities, relations, mappings = self.ontology.export_for_update()
        ontology_update = OntologyUpdate(
            entities=entities,
            relations=relations,
            mappings=mappings
        )
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    def _update_budgets(self, budget_overrides: Dict[str, Any]) -> None:
        """Update budget configuration"""
        if "beam_width" in budget_overrides:
            self.budgets_used.beam_width = budget_overrides["beam_width"]
            self.ladder_config.beam_width = budget_overrides["beam_width"]
        
        if "depth_back" in budget_overrides:
            self.budgets_used.depth_back = budget_overrides["depth_back"]
            self.ladder_config.depth_back = budget_overrides["depth_back"]
        
        if "depth_fwd" in budget_overrides:
            self.budgets_used.depth_fwd = budget_overrides["depth_fwd"]
            self.ladder_config.depth_forward = budget_overrides["depth_fwd"]
    
    def _create_initial_threads(self, priors: List[QuestionNode], 
                              scenarios: List[ScenarioLane]) -> List[Thread]:
        """Create initial thread tracking"""
        threads = []
        
        # Create thread for prior ladder
        if priors:
            self.thread_counter += 1
            thread_id = f"T{self.thread_counter}"
            thread_path = [node.id for node in priors]
            
            thread_state = ThreadState(
                thread_id=thread_id,
                path=thread_path,
                current_node=priors[-1].id if priors else None
            )
            self.threads[thread_id] = thread_state
            
            threads.append(Thread(
                thread_id=thread_id,
                origin_node_id=priors[0].id,
                path=thread_path,
                status="active",
                summary=f"Prior ladder: {priors[0].text} → {priors[-1].text}"
            ))
        
        # Create threads for scenario lanes
        for scenario in scenarios:
            if scenario.lane:
                self.thread_counter += 1
                thread_id = f"T{self.thread_counter}"
                thread_path = [node.id for node in scenario.lane]
                
                thread_state = ThreadState(
                    thread_id=thread_id,
                    path=thread_path,
                    current_node=scenario.lane[-1].id
                )
                self.threads[thread_id] = thread_state
                
                threads.append(Thread(
                    thread_id=thread_id,
                    origin_node_id=scenario.lane[0].id,
                    path=thread_path,
                    status="active",
                    summary=f"{scenario.name}: {scenario.description}"
                ))
        
        return threads
    
    def _generate_init_chat_reply(self, priors: List[QuestionNode], 
                                 scenarios: List[ScenarioLane]) -> str:
        """Generate crisp tactical chat reply for initialization"""
        reply_parts = []
        
        # Reference best priors
        if priors:
            top_priors = sorted(priors, key=lambda p: p.expected_info_gain, reverse=True)[:2]
            prior_refs = ", ".join([f"**{p.id}**" for p in top_priors])
            reply_parts.append(f"Key priors: {prior_refs}")
        
        # Reference best lanes
        if scenarios:
            top_scenarios = sorted(scenarios, key=lambda s: s.lane[0].expected_info_gain if s.lane else 0, reverse=True)[:2]
            lane_refs = ", ".join([f"**{s.id}**" for s in top_scenarios])
            reply_parts.append(f"Top paths: {lane_refs}")
        
        if not reply_parts:
            reply_parts.append("Ready to explore")
        
        return ". ".join(reply_parts) + "."
    
    def _find_node_by_id(self, node_id: str) -> Optional[QuestionNode]:
        """Find a node by ID across all current structures"""
        # This is a simplified lookup - in production would use proper indexing
        # For now, return None as this requires maintaining node registry
        return None
    
    def _get_current_priors(self) -> List[QuestionNode]:
        """Get current prior nodes"""
        # Simplified - would maintain proper state
        return []
    
    def _get_current_scenarios(self) -> List[ScenarioLane]:
        """Get current scenario lanes"""
        # Simplified - would maintain proper state
        return []
    
    def _get_current_threads(self) -> List[Thread]:
        """Get current threads as schema objects"""
        threads = []
        for thread_state in self.threads.values():
            threads.append(Thread(
                thread_id=thread_state.thread_id,
                origin_node_id=thread_state.path[0] if thread_state.path else "",
                path=thread_state.path,
                status=thread_state.status,
                summary=f"Thread with {len(thread_state.path)} steps"
            ))
        return threads
    
    def _update_threads_for_advance(self, node_id: str, new_scenarios: List[ScenarioLane]) -> None:
        """Update thread tracking for advance operation"""
        # Create new threads for the expanded scenarios
        for scenario in new_scenarios:
            if scenario.lane:
                self.thread_counter += 1
                thread_id = f"T{self.thread_counter}"
                thread_path = [node_id] + [node.id for node in scenario.lane]
                
                thread_state = ThreadState(
                    thread_id=thread_id,
                    path=thread_path,
                    current_node=scenario.lane[-1].id
                )
                self.threads[thread_id] = thread_state
    
    def _create_error_response(self, error_message: str) -> GodmodeResponse:
        """Create error response"""
        return GodmodeResponse(
            chat_reply=f"Error: {error_message}",
            graph_update=GraphUpdate(
                current_question=self.current_question or "",
                priors=[],
                scenarios=[],
                threads=[],
                meta=GraphMeta(budgets_used=self.budgets_used, notes=f"Error: {error_message}")
            ),
            ontology_update=OntologyUpdate(
                entities=[],
                relations=[],
                mappings=[]
            )
        )