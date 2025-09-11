"""
Forward reasoning engine for goal-directed problem solving.

This module implements forward chaining reasoning that starts from
known facts and works towards conclusions.
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


class ForwardReasoningEngine:
    """
    Forward reasoning engine implementing goal-directed reasoning.
    
    This engine starts from initial facts and applies rules to derive
    new facts until a solution is found or no more progress can be made.
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
        self.current_facts: Set[str] = set()
        self.derived_facts: Set[str] = set()
        self.applied_rules: List[Dict[str, Any]] = []
        
    async def reason(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        max_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute forward reasoning to solve the problem.
        
        Args:
            problem: Problem to solve
            cognitive_state: Current cognitive state
            max_steps: Maximum reasoning steps
            
        Returns:
            Dict containing reasoning results
        """
        logger.info(f"Starting forward reasoning for problem: {problem.title}")
        
        # Initialize facts from problem
        self.current_facts = self._extract_initial_facts(problem)
        self.derived_facts = set()
        self.applied_rules = []
        
        steps = []
        solutions = []
        
        # Forward chaining loop
        for step_num in range(max_steps):
            # Check if we can derive new facts
            new_facts, applied_rule = await self._apply_forward_rules(
                self.current_facts, problem
            )
            
            if not new_facts:
                logger.info("No new facts can be derived, stopping")
                break
            
            # Create reasoning step
            reasoning_step = ReasoningStep(
                step_number=step_num,
                operation="forward_inference",
                input_state={"current_facts": list(self.current_facts)},
                output_state={"new_facts": list(new_facts)},
                cognitive_level=CognitiveLevel.OPERATIONAL,
                rationale=f"Applied rule: {applied_rule.get('name', 'unknown')}",
                confidence=applied_rule.get('confidence', 0.7),
            )
            steps.append(reasoning_step)
            
            # Update facts
            self.current_facts.update(new_facts)
            self.derived_facts.update(new_facts)
            self.applied_rules.append(applied_rule)
            
            # Check if we have a solution
            solution = await self._check_for_solution(problem, self.current_facts)
            if solution:
                solutions.append(solution)
                logger.info(f"Solution found at step {step_num}")
                break
            
            # Store intermediate results in memory
            await self.memory.store({
                "type": "forward_reasoning_step",
                "step": step_num,
                "facts": list(self.current_facts),
                "rule_applied": applied_rule,
            })
        
        # Generate final solution if none found during reasoning
        if not solutions:
            final_solution = await self._generate_final_solution(problem, self.current_facts)
            if final_solution:
                solutions.append(final_solution)
        
        return {
            "steps": steps,
            "solutions": solutions,
            "final_facts": list(self.current_facts),
            "derived_facts": list(self.derived_facts),
            "rules_applied": len(self.applied_rules),
        }
    
    def _extract_initial_facts(self, problem: Problem) -> Set[str]:
        """Extract initial facts from the problem description."""
        facts = set()
        
        # Extract facts from problem description
        words = problem.description.lower().split()
        
        # Simple fact extraction based on keywords
        fact_patterns = [
            ("has", "possession"),
            ("is", "property"),
            ("can", "capability"),
            ("must", "requirement"),
            ("should", "preference"),
            ("needs", "need"),
            ("wants", "desire"),
        ]
        
        for i, word in enumerate(words):
            for pattern, fact_type in fact_patterns:
                if word == pattern and i < len(words) - 1:
                    # Create fact from pattern
                    if i > 0 and i < len(words) - 1:
                        subject = words[i-1]
                        object_part = " ".join(words[i+1:i+3])  # Take next 1-2 words
                        fact = f"{subject}_{pattern}_{object_part.replace(' ', '_')}"
                        facts.add(fact)
        
        # Add problem constraints as facts
        for constraint in problem.constraints:
            constraint_fact = f"constraint_{constraint.lower().replace(' ', '_')}"
            facts.add(constraint_fact)
        
        # Add problem objectives as goals
        for objective in problem.objectives:
            objective_fact = f"goal_{objective.lower().replace(' ', '_')}"
            facts.add(objective_fact)
        
        # Add domain-specific facts
        domain_facts = self._get_domain_facts(problem.domain)
        facts.update(domain_facts)
        
        logger.info(f"Extracted {len(facts)} initial facts")
        return facts
    
    def _get_domain_facts(self, domain: str) -> Set[str]:
        """Get domain-specific facts."""
        domain_knowledge = {
            "logistics": {
                "transportation_has_cost",
                "distance_affects_time",
                "capacity_limits_load",
                "fuel_required_for_movement",
                "routes_can_be_optimized",
            },
            "planning": {
                "goals_require_actions",
                "actions_have_preconditions",
                "resources_are_limited",
                "time_is_constraint",
                "plans_can_fail",
            },
            "optimization": {
                "objectives_can_conflict",
                "constraints_limit_solutions",
                "tradeoffs_exist",
                "local_optima_possible",
                "global_optimum_desired",
            },
            "general": {
                "problems_have_solutions",
                "solutions_require_evaluation",
                "resources_are_finite",
                "time_progresses_forward",
            }
        }
        
        return domain_knowledge.get(domain, domain_knowledge["general"])
    
    async def _apply_forward_rules(
        self,
        current_facts: Set[str],
        problem: Problem,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """Apply forward reasoning rules to derive new facts."""
        
        new_facts = set()
        applied_rule = {"name": "none", "confidence": 0.0}
        
        # Rule 1: Composition rule
        composition_facts = self._apply_composition_rule(current_facts)
        if composition_facts:
            new_facts.update(composition_facts)
            applied_rule = {"name": "composition", "confidence": 0.8}
        
        # Rule 2: Implication rule
        if not new_facts:
            implication_facts = self._apply_implication_rule(current_facts)
            if implication_facts:
                new_facts.update(implication_facts)
                applied_rule = {"name": "implication", "confidence": 0.7}
        
        # Rule 3: Domain-specific rules
        if not new_facts:
            domain_facts = await self._apply_domain_rules(current_facts, problem.domain)
            if domain_facts:
                new_facts.update(domain_facts)
                applied_rule = {"name": "domain_specific", "confidence": 0.6}
        
        # Rule 4: Memory-based rules
        if not new_facts:
            memory_facts = await self._apply_memory_rules(current_facts)
            if memory_facts:
                new_facts.update(memory_facts)
                applied_rule = {"name": "memory_based", "confidence": 0.5}
        
        return new_facts, applied_rule
    
    def _apply_composition_rule(self, facts: Set[str]) -> Set[str]:
        """Apply composition rules to combine facts."""
        new_facts = set()
        
        # Look for facts that can be composed
        fact_list = list(facts)
        
        for i, fact1 in enumerate(fact_list):
            for j, fact2 in enumerate(fact_list[i+1:], i+1):
                # Simple composition: if A has B and B has C, then A has C
                if "_has_" in fact1 and "_has_" in fact2:
                    parts1 = fact1.split("_has_")
                    parts2 = fact2.split("_has_")
                    
                    if len(parts1) == 2 and len(parts2) == 2:
                        if parts1[1] == parts2[0]:  # B matches
                            composed_fact = f"{parts1[0]}_has_{parts2[1]}"
                            new_facts.add(composed_fact)
                
                # Causality composition: if A causes B and B causes C, then A causes C
                if "_causes_" in fact1 and "_causes_" in fact2:
                    parts1 = fact1.split("_causes_")
                    parts2 = fact2.split("_causes_")
                    
                    if len(parts1) == 2 and len(parts2) == 2:
                        if parts1[1] == parts2[0]:
                            composed_fact = f"{parts1[0]}_causes_{parts2[1]}"
                            new_facts.add(composed_fact)
        
        return new_facts
    
    def _apply_implication_rule(self, facts: Set[str]) -> Set[str]:
        """Apply implication rules to derive new facts."""
        new_facts = set()
        
        # Define implication patterns
        implications = [
            # If something is required, then it's needed
            ("_requires_", "_needs_"),
            # If something is possible, then it can be done
            ("_possible_", "_can_be_"),
            # If something is optimal, then it's good
            ("_optimal_", "_good_"),
            # If something is constrained, then it's limited
            ("_constraint_", "_limited_"),
        ]
        
        for fact in facts:
            for pattern, conclusion in implications:
                if pattern in fact:
                    new_fact = fact.replace(pattern, conclusion)
                    new_facts.add(new_fact)
        
        return new_facts
    
    async def _apply_domain_rules(self, facts: Set[str], domain: str) -> Set[str]:
        """Apply domain-specific reasoning rules."""
        new_facts = set()
        
        if domain == "logistics":
            # Logistics-specific rules
            for fact in facts:
                if "transportation" in fact and "cost" in fact:
                    new_facts.add("budget_constraint_exists")
                if "distance" in fact and "time" in fact:
                    new_facts.add("speed_optimization_possible")
                if "capacity" in fact and "load" in fact:
                    new_facts.add("resource_allocation_needed")
        
        elif domain == "planning":
            # Planning-specific rules
            for fact in facts:
                if "goal_" in fact:
                    goal_name = fact.replace("goal_", "")
                    new_facts.add(f"action_needed_for_{goal_name}")
                if "constraint_" in fact:
                    constraint_name = fact.replace("constraint_", "")
                    new_facts.add(f"limitation_exists_{constraint_name}")
        
        elif domain == "optimization":
            # Optimization-specific rules
            for fact in facts:
                if "objective" in fact:
                    new_facts.add("optimization_problem_exists")
                if "constraint" in fact:
                    new_facts.add("feasible_region_limited")
                if "tradeoff" in fact:
                    new_facts.add("pareto_optimization_needed")
        
        return new_facts
    
    async def _apply_memory_rules(self, facts: Set[str]) -> Set[str]:
        """Apply rules based on memory retrieval."""
        new_facts = set()
        
        # Query memory for related information
        for fact in list(facts)[:5]:  # Limit to avoid too many queries
            # Extract key terms from fact
            terms = fact.replace("_", " ").split()
            query = " ".join(terms[:3])  # Use first 3 terms
            
            # Retrieve related memories
            memory_results = await self.memory.retrieve(query, top_k=2)
            
            for memory_item in memory_results:
                content = memory_item.content
                if isinstance(content, dict) and "derived_fact" in content:
                    new_facts.add(content["derived_fact"])
        
        return new_facts
    
    async def _check_for_solution(
        self,
        problem: Problem,
        current_facts: Set[str],
    ) -> Optional[Solution]:
        """Check if current facts constitute a solution."""
        
        # Check if we have facts that address the problem objectives
        objectives_addressed = 0
        solution_elements = []
        
        for objective in problem.objectives:
            objective_terms = objective.lower().replace(" ", "_")
            
            # Look for facts that relate to this objective
            for fact in current_facts:
                if any(term in fact for term in objective_terms.split("_")):
                    objectives_addressed += 1
                    solution_elements.append(f"Objective '{objective}' addressed by: {fact}")
                    break
        
        # If we've addressed most objectives, create a solution
        if objectives_addressed >= len(problem.objectives) * 0.7:  # 70% threshold
            solution_text = "Forward reasoning solution:\n"
            solution_text += "\n".join(solution_elements)
            
            # Add derived insights
            if self.derived_facts:
                solution_text += "\n\nKey insights derived:"
                for fact in list(self.derived_facts)[:5]:  # Top 5 insights
                    readable_fact = fact.replace("_", " ").title()
                    solution_text += f"\n- {readable_fact}"
            
            confidence = min(0.9, objectives_addressed / max(1, len(problem.objectives)))
            
            solution = Solution(
                problem_id=problem.id,
                solution_text=solution_text,
                confidence=confidence,
                completeness=objectives_addressed / max(1, len(problem.objectives)),
                feasibility=0.7,  # Assume reasonable feasibility
                solving_time=None,  # Would be computed by caller
                reasoning_steps=len(self.applied_rules),
            )
            
            return solution
        
        return None
    
    async def _generate_final_solution(
        self,
        problem: Problem,
        current_facts: Set[str],
    ) -> Optional[Solution]:
        """Generate a final solution based on all derived facts."""
        
        if not current_facts:
            return None
        
        # Create solution from most relevant facts
        relevant_facts = []
        
        # Filter facts that seem most relevant to the problem
        problem_terms = set(problem.description.lower().split())
        
        for fact in current_facts:
            fact_terms = set(fact.replace("_", " ").split())
            overlap = len(problem_terms & fact_terms)
            
            if overlap > 0:
                relevant_facts.append((fact, overlap))
        
        # Sort by relevance
        relevant_facts.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_facts:
            solution_text = f"Forward reasoning analysis of '{problem.title}':\n\n"
            
            solution_text += "Key findings:\n"
            for fact, _ in relevant_facts[:7]:  # Top 7 most relevant facts
                readable_fact = fact.replace("_", " ").title()
                solution_text += f"• {readable_fact}\n"
            
            if self.applied_rules:
                solution_text += f"\nReasoning process applied {len(self.applied_rules)} inference rules "
                solution_text += f"and derived {len(self.derived_facts)} new insights."
            
            # Estimate confidence based on fact relevance and coverage
            confidence = min(0.7, len(relevant_facts) / 10.0)
            
            solution = Solution(
                problem_id=problem.id,
                solution_text=solution_text,
                confidence=confidence,
                completeness=0.6,  # Partial solution
                feasibility=0.6,
                solving_time=None,
                reasoning_steps=len(self.applied_rules),
            )
            
            return solution
        
        return None