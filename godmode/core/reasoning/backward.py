"""
Backward reasoning engine for goal-oriented problem solving.

This module implements backward chaining reasoning that starts from
goals and works backwards to find supporting facts and actions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from godmode.models.core import (
    Problem,
    Solution,
    CognitiveState,
    ReasoningStep,
    CognitiveLevel,
)
from godmode.core.memory import CognitiveMemory


logger = logging.getLogger(__name__)


class BackwardReasoningEngine:
    """
    Backward reasoning engine implementing goal-oriented reasoning.
    
    This engine starts from desired goals and works backwards to find
    the facts, conditions, and actions needed to achieve those goals.
    """
    
    def __init__(
        self,
        memory: CognitiveMemory,
        max_depth: int = 10,
        use_gpu: bool = False,
    ):
        self.memory = memory
        self.max_depth = max_depth
        self.use_gpu = use_gpu
        
        # Reasoning state
        self.goals_stack: List[str] = []
        self.achieved_goals: Set[str] = set()
        self.required_conditions: Dict[str, List[str]] = {}
        self.solution_path: List[Dict[str, Any]] = []
        
    async def reason(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        max_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute backward reasoning to solve the problem.
        
        Args:
            problem: Problem to solve
            cognitive_state: Current cognitive state
            max_steps: Maximum reasoning steps
            
        Returns:
            Dict containing reasoning results
        """
        logger.info(f"Starting backward reasoning for problem: {problem.title}")
        
        # Initialize goals from problem objectives
        self.goals_stack = self._extract_goals(problem)
        self.achieved_goals = set()
        self.required_conditions = {}
        self.solution_path = []
        
        steps = []
        solutions = []
        
        # Backward chaining loop
        for step_num in range(max_steps):
            if not self.goals_stack:
                logger.info("All goals processed")
                break
            
            # Get current goal
            current_goal = self.goals_stack.pop()
            
            # Check if goal is already achieved
            if current_goal in self.achieved_goals:
                continue
            
            # Find requirements for achieving this goal
            requirements = await self._find_goal_requirements(current_goal, problem)
            
            # Create reasoning step
            reasoning_step = ReasoningStep(
                step_number=step_num,
                operation="backward_inference",
                input_state={"current_goal": current_goal},
                output_state={"requirements": requirements},
                cognitive_level=CognitiveLevel.EXECUTIVE,
                rationale=f"Finding requirements for goal: {current_goal}",
                confidence=0.7,
            )
            steps.append(reasoning_step)
            
            # Process requirements
            if requirements:
                self.required_conditions[current_goal] = requirements
                
                # Add unmet requirements as new subgoals
                for req in requirements:
                    if not await self._is_condition_met(req, problem):
                        if req not in self.goals_stack:
                            self.goals_stack.append(req)
                
                # Record solution step
                self.solution_path.append({
                    "goal": current_goal,
                    "requirements": requirements,
                    "step": step_num,
                })
            else:
                # Goal can be achieved directly
                self.achieved_goals.add(current_goal)
                self.solution_path.append({
                    "goal": current_goal,
                    "requirements": [],
                    "step": step_num,
                    "achievable": True,
                })
            
            # Store reasoning step in memory
            await self.memory.store({
                "type": "backward_reasoning_step",
                "step": step_num,
                "goal": current_goal,
                "requirements": requirements,
            })
            
            # Check if we can construct a solution
            if len(self.solution_path) >= 3:  # Minimum path length
                solution = await self._construct_solution(problem)
                if solution:
                    solutions.append(solution)
                    break
        
        # Generate final solution if none found during reasoning
        if not solutions and self.solution_path:
            final_solution = await self._generate_final_solution(problem)
            if final_solution:
                solutions.append(final_solution)
        
        return {
            "steps": steps,
            "solutions": solutions,
            "solution_path": self.solution_path,
            "achieved_goals": list(self.achieved_goals),
            "remaining_goals": self.goals_stack,
        }
    
    def _extract_goals(self, problem: Problem) -> List[str]:
        """Extract goals from problem objectives and description."""
        goals = []
        
        # Add explicit objectives as goals
        for objective in problem.objectives:
            goal = f"achieve_{objective.lower().replace(' ', '_')}"
            goals.append(goal)
        
        # Extract implicit goals from problem description
        goal_keywords = [
            "solve", "find", "optimize", "minimize", "maximize", 
            "improve", "reduce", "increase", "create", "design",
            "plan", "develop", "implement", "achieve"
        ]
        
        words = problem.description.lower().split()
        for i, word in enumerate(words):
            if word in goal_keywords and i < len(words) - 1:
                # Create goal from keyword and following words
                goal_phrase = " ".join(words[i:i+3])  # Take 3 words
                goal = f"goal_{goal_phrase.replace(' ', '_')}"
                goals.append(goal)
        
        # Add meta-goal
        goals.append(f"solve_problem_{problem.problem_type}")
        
        # Reverse to process most specific goals first
        goals.reverse()
        
        logger.info(f"Extracted {len(goals)} goals for backward reasoning")
        return goals
    
    async def _find_goal_requirements(
        self,
        goal: str,
        problem: Problem,
    ) -> List[str]:
        """Find requirements needed to achieve a goal."""
        requirements = []
        
        # Domain-specific requirements
        domain_reqs = self._get_domain_requirements(goal, problem.domain)
        requirements.extend(domain_reqs)
        
        # Pattern-based requirements
        pattern_reqs = self._get_pattern_requirements(goal)
        requirements.extend(pattern_reqs)
        
        # Memory-based requirements
        memory_reqs = await self._get_memory_requirements(goal)
        requirements.extend(memory_reqs)
        
        # Problem-specific requirements
        problem_reqs = self._get_problem_requirements(goal, problem)
        requirements.extend(problem_reqs)
        
        # Remove duplicates
        requirements = list(set(requirements))
        
        return requirements[:5]  # Limit to top 5 requirements
    
    def _get_domain_requirements(self, goal: str, domain: str) -> List[str]:
        """Get domain-specific requirements for achieving a goal."""
        requirements = []
        
        if domain == "logistics":
            if "transport" in goal or "move" in goal:
                requirements.extend([
                    "have_vehicle_available",
                    "know_route_information", 
                    "have_sufficient_fuel",
                    "verify_capacity_limits"
                ])
            elif "optimize" in goal:
                requirements.extend([
                    "define_optimization_criteria",
                    "identify_constraints",
                    "collect_performance_data"
                ])
        
        elif domain == "planning":
            if "plan" in goal or "schedule" in goal:
                requirements.extend([
                    "identify_all_tasks",
                    "determine_dependencies",
                    "estimate_task_durations",
                    "allocate_resources"
                ])
            elif "achieve" in goal:
                requirements.extend([
                    "break_down_into_steps",
                    "identify_prerequisites",
                    "verify_feasibility"
                ])
        
        elif domain == "optimization":
            if "minimize" in goal or "maximize" in goal:
                requirements.extend([
                    "define_objective_function",
                    "identify_decision_variables",
                    "specify_constraints",
                    "choose_optimization_method"
                ])
        
        # General requirements
        if "solve" in goal:
            requirements.extend([
                "understand_problem_fully",
                "gather_relevant_information",
                "consider_alternative_approaches"
            ])
        
        return requirements
    
    def _get_pattern_requirements(self, goal: str) -> List[str]:
        """Get requirements based on common goal patterns."""
        requirements = []
        
        # Action-based patterns
        action_patterns = {
            "create": ["gather_materials", "have_design", "allocate_time"],
            "design": ["understand_requirements", "research_options", "evaluate_constraints"],
            "implement": ["have_plan", "allocate_resources", "verify_prerequisites"],
            "optimize": ["define_metrics", "identify_variables", "set_constraints"],
            "reduce": ["measure_current_state", "identify_causes", "plan_interventions"],
            "increase": ["understand_current_level", "identify_growth_factors", "remove_barriers"],
        }
        
        for action, reqs in action_patterns.items():
            if action in goal:
                requirements.extend(reqs)
                break
        
        # Object-based patterns
        if "system" in goal:
            requirements.extend(["define_system_boundaries", "identify_components", "understand_interactions"])
        
        if "process" in goal:
            requirements.extend(["map_current_process", "identify_bottlenecks", "define_improvements"])
        
        return requirements
    
    async def _get_memory_requirements(self, goal: str) -> List[str]:
        """Get requirements based on memory retrieval."""
        requirements = []
        
        # Query memory for similar goals
        goal_terms = goal.replace("_", " ")
        memory_results = await self.memory.retrieve(goal_terms, top_k=3)
        
        for memory_item in memory_results:
            content = memory_item.content
            
            # Extract requirements from memory
            if isinstance(content, dict):
                if "requirements" in content:
                    if isinstance(content["requirements"], list):
                        requirements.extend(content["requirements"][:2])  # Take first 2
                
                if "prerequisites" in content:
                    if isinstance(content["prerequisites"], list):
                        requirements.extend(content["prerequisites"][:2])
        
        return requirements
    
    def _get_problem_requirements(self, goal: str, problem: Problem) -> List[str]:
        """Get requirements specific to the current problem."""
        requirements = []
        
        # Requirements based on problem constraints
        for constraint in problem.constraints:
            if any(word in goal for word in constraint.lower().split()):
                req = f"satisfy_constraint_{constraint.lower().replace(' ', '_')}"
                requirements.append(req)
        
        # Requirements based on problem domain
        if problem.domain in goal or goal.endswith(problem.domain):
            requirements.append(f"apply_{problem.domain}_knowledge")
            requirements.append(f"use_{problem.domain}_methods")
        
        # Requirements based on problem complexity
        if problem.complexity == "high":
            requirements.extend([
                "break_into_subproblems",
                "use_systematic_approach",
                "validate_intermediate_results"
            ])
        elif problem.complexity == "medium":
            requirements.extend([
                "plan_solution_approach",
                "gather_necessary_information"
            ])
        
        return requirements
    
    async def _is_condition_met(self, condition: str, problem: Problem) -> bool:
        """Check if a condition is already met."""
        
        # Check against problem description and constraints
        problem_text = f"{problem.description} {' '.join(problem.constraints)}".lower()
        condition_words = condition.replace("_", " ").split()
        
        # Simple heuristic: condition is met if most words appear in problem
        matches = sum(1 for word in condition_words if word in problem_text)
        return matches >= len(condition_words) * 0.6
    
    async def _construct_solution(self, problem: Problem) -> Optional[Solution]:
        """Construct solution from the current solution path."""
        
        if len(self.solution_path) < 2:
            return None
        
        # Build solution text from path
        solution_text = f"Backward reasoning solution for '{problem.title}':\n\n"
        
        # Organize path by goals and requirements
        solution_text += "Goal Achievement Plan:\n"
        
        for i, step in enumerate(reversed(self.solution_path), 1):
            goal = step["goal"].replace("_", " ").title()
            solution_text += f"\n{i}. {goal}\n"
            
            if step["requirements"]:
                solution_text += "   Requirements:\n"
                for req in step["requirements"]:
                    req_text = req.replace("_", " ").title()
                    solution_text += f"   • {req_text}\n"
            else:
                solution_text += "   • Can be achieved directly\n"
        
        # Add summary
        total_goals = len(self.solution_path)
        achieved_goals = len(self.achieved_goals)
        
        solution_text += f"\nSummary: Identified {total_goals} goals with detailed requirements. "
        solution_text += f"Direct achievement possible for {achieved_goals} goals."
        
        # Calculate confidence based on goal coverage and requirement depth
        confidence = min(0.8, (achieved_goals + len(self.solution_path) * 0.1) / max(1, total_goals))
        
        solution = Solution(
            problem_id=problem.id,
            solution_text=solution_text,
            confidence=confidence,
            completeness=len(self.solution_path) / max(1, len(problem.objectives) + 2),
            feasibility=0.7,  # Backward reasoning tends to be feasible
            solving_time=None,
            reasoning_steps=len(self.solution_path),
        )
        
        return solution
    
    async def _generate_final_solution(self, problem: Problem) -> Optional[Solution]:
        """Generate final solution from accumulated reasoning."""
        
        if not self.solution_path:
            return None
        
        solution_text = f"Backward analysis of '{problem.title}':\n\n"
        
        # Summarize the goal decomposition
        solution_text += "Goal Decomposition Analysis:\n"
        
        main_goals = [step for step in self.solution_path if not step["goal"].startswith("goal_")]
        sub_goals = [step for step in self.solution_path if step["goal"].startswith("goal_")]
        
        if main_goals:
            solution_text += f"\nPrimary objectives ({len(main_goals)}):\n"
            for step in main_goals[:5]:  # Top 5
                goal_name = step["goal"].replace("_", " ").title()
                req_count = len(step.get("requirements", []))
                solution_text += f"• {goal_name} ({req_count} requirements)\n"
        
        if sub_goals:
            solution_text += f"\nSupporting goals ({len(sub_goals)}):\n"
            for step in sub_goals[:3]:  # Top 3
                goal_name = step["goal"].replace("goal_", "").replace("_", " ").title()
                solution_text += f"• {goal_name}\n"
        
        # Add key insights
        all_requirements = []
        for step in self.solution_path:
            all_requirements.extend(step.get("requirements", []))
        
        if all_requirements:
            # Find most common requirements
            req_counts = {}
            for req in all_requirements:
                req_counts[req] = req_counts.get(req, 0) + 1
            
            common_reqs = sorted(req_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            solution_text += f"\nKey requirements identified:\n"
            for req, count in common_reqs:
                req_text = req.replace("_", " ").title()
                solution_text += f"• {req_text} (mentioned {count} times)\n"
        
        # Estimate solution quality
        confidence = min(0.6, len(self.solution_path) / 8.0)  # Scale with path length
        
        solution = Solution(
            problem_id=problem.id,
            solution_text=solution_text,
            confidence=confidence,
            completeness=0.5,  # Partial analysis
            feasibility=0.6,
            solving_time=None,
            reasoning_steps=len(self.solution_path),
        )
        
        return solution
