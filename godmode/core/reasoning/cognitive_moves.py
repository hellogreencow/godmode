"""
Cognitive moves engine implementing flexible reasoning strategies.

This module implements various cognitive moves and heuristics for
problem-solving, including analogical reasoning, abstraction, and
creative problem-solving techniques.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from godmode.models.core import (
    Problem,
    Solution,
    CognitiveState,
    ReasoningStep,
    CognitiveLevel,
)
from godmode.core.memory import CognitiveMemory


logger = logging.getLogger(__name__)


class CognitiveMoveEngine:
    """
    Cognitive moves engine implementing flexible reasoning strategies.
    
    This engine applies various cognitive moves and heuristics to
    approach problems from different angles and generate creative solutions.
    """
    
    def __init__(
        self,
        memory: CognitiveMemory,
        use_gpu: bool = False,
    ):
        self.memory = memory
        self.use_gpu = use_gpu
        
        # Available cognitive moves
        self.cognitive_moves = [
            self._analogical_reasoning,
            self._abstraction_move,
            self._decomposition_move,
            self._perspective_shift,
            self._constraint_relaxation,
            self._resource_substitution,
            self._temporal_reasoning,
            self._causal_analysis,
        ]
        
        # Move history for this session
        self.move_history: List[Dict[str, Any]] = []
        
    async def reason(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        max_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute cognitive moves reasoning to solve the problem.
        
        Args:
            problem: Problem to solve
            cognitive_state: Current cognitive state
            max_steps: Maximum reasoning steps
            
        Returns:
            Dict containing reasoning results
        """
        logger.info(f"Starting cognitive moves reasoning for problem: {problem.title}")
        
        self.move_history = []
        steps = []
        solutions = []
        insights = []
        
        # Apply different cognitive moves
        for step_num in range(max_steps):
            # Select a cognitive move to apply
            move_func = self._select_cognitive_move(step_num, problem)
            
            # Apply the cognitive move
            move_result = await move_func(problem, cognitive_state, step_num)
            
            if not move_result:
                continue
            
            # Create reasoning step
            reasoning_step = ReasoningStep(
                step_number=step_num,
                operation=move_result["move_name"],
                input_state=move_result.get("input_state", {}),
                output_state=move_result.get("output_state", {}),
                cognitive_level=move_result.get("cognitive_level", CognitiveLevel.OPERATIONAL),
                rationale=move_result.get("rationale", "Applied cognitive move"),
                confidence=move_result.get("confidence", 0.5),
            )
            steps.append(reasoning_step)
            
            # Collect insights
            if "insights" in move_result:
                insights.extend(move_result["insights"])
            
            # Check for solutions
            if "solution" in move_result:
                solutions.append(move_result["solution"])
            
            # Store move in history
            self.move_history.append({
                "step": step_num,
                "move": move_result["move_name"],
                "success": move_result.get("success", False),
                "insights_generated": len(move_result.get("insights", [])),
            })
            
            # Store in memory
            await self.memory.store({
                "type": "cognitive_move",
                "move_name": move_result["move_name"],
                "problem_domain": problem.domain,
                "insights": move_result.get("insights", []),
                "success": move_result.get("success", False),
            })
        
        # Generate final solution if none found
        if not solutions and insights:
            final_solution = await self._synthesize_solution(problem, insights)
            if final_solution:
                solutions.append(final_solution)
        
        return {
            "steps": steps,
            "solutions": solutions,
            "insights": insights,
            "moves_applied": [move["move"] for move in self.move_history],
            "success_rate": sum(1 for move in self.move_history if move["success"]) / max(1, len(self.move_history)),
        }
    
    def _select_cognitive_move(self, step_num: int, problem: Problem) -> Any:
        """Select the most appropriate cognitive move for the current step."""
        
        # Early steps: use broad, exploratory moves
        if step_num < 3:
            early_moves = [
                self._analogical_reasoning,
                self._abstraction_move,
                self._perspective_shift,
            ]
            return random.choice(early_moves)
        
        # Middle steps: use analytical moves
        elif step_num < 7:
            analytical_moves = [
                self._decomposition_move,
                self._causal_analysis,
                self._constraint_relaxation,
            ]
            return random.choice(analytical_moves)
        
        # Later steps: use synthetic moves
        else:
            synthetic_moves = [
                self._resource_substitution,
                self._temporal_reasoning,
                self._abstraction_move,  # Can be used for synthesis too
            ]
            return random.choice(synthetic_moves)
    
    async def _analogical_reasoning(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Apply analogical reasoning to find similar problems and solutions."""
        
        # Query memory for analogous problems
        query = f"{problem.domain} {problem.problem_type}"
        similar_problems = await self.memory.retrieve(query, top_k=3)
        
        insights = []
        analogies = []
        
        for memory_item in similar_problems:
            content = memory_item.content
            
            if isinstance(content, dict) and "problem" in content:
                # Extract analogy
                analogy = {
                    "source": content.get("problem", "unknown"),
                    "similarity": memory_item.get_activation(),
                    "applicable_solution": content.get("solution", ""),
                }
                analogies.append(analogy)
                
                # Generate insight
                insight = f"Similar to {analogy['source']}: {analogy['applicable_solution'][:100]}"
                insights.append(insight)
        
        # Look for structural similarities
        if problem.problem_type == "optimization":
            insights.append("This is an optimization problem - consider gradient descent, genetic algorithms, or linear programming approaches")
        elif problem.problem_type == "planning":
            insights.append("This is a planning problem - consider hierarchical task networks, A* search, or constraint satisfaction")
        elif problem.problem_type == "design":
            insights.append("This is a design problem - consider iterative refinement, user-centered design, or biomimicry")
        
        # Domain analogies
        domain_analogies = {
            "logistics": "Similar to supply chain optimization or transportation problems",
            "planning": "Similar to project management or resource allocation problems", 
            "optimization": "Similar to mathematical programming or operations research problems",
        }
        
        if problem.domain in domain_analogies:
            insights.append(domain_analogies[problem.domain])
        
        return {
            "move_name": "analogical_reasoning",
            "input_state": {"query": query},
            "output_state": {"analogies": analogies},
            "cognitive_level": CognitiveLevel.METACOGNITIVE,
            "rationale": "Finding analogous problems and solutions",
            "confidence": 0.6,
            "insights": insights,
            "success": len(insights) > 0,
        }
    
    async def _abstraction_move(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Apply abstraction to identify higher-level patterns and principles."""
        
        insights = []
        abstractions = []
        
        # Abstract the problem to higher levels
        problem_elements = problem.description.lower().split()
        
        # Identify abstract categories
        abstract_categories = {
            "optimization": ["minimize", "maximize", "optimize", "best", "efficient"],
            "allocation": ["distribute", "allocate", "assign", "share", "divide"],
            "transformation": ["change", "convert", "transform", "modify", "improve"],
            "coordination": ["coordinate", "synchronize", "align", "integrate", "combine"],
            "selection": ["choose", "select", "pick", "decide", "determine"],
        }
        
        for category, keywords in abstract_categories.items():
            if any(keyword in problem_elements for keyword in keywords):
                abstractions.append(category)
                insights.append(f"This problem involves {category} - consider general {category} principles")
        
        # Abstract to mathematical formulation
        if any(word in problem_elements for word in ["number", "amount", "quantity", "rate", "cost"]):
            insights.append("This problem has quantitative aspects - consider mathematical modeling")
            abstractions.append("quantitative_problem")
        
        # Abstract to system thinking
        if any(word in problem_elements for word in ["system", "process", "workflow", "interaction"]):
            insights.append("This problem involves systems - consider systems thinking approaches")
            abstractions.append("systems_problem")
        
        # Abstract to constraint satisfaction
        if problem.constraints or any(word in problem_elements for word in ["must", "cannot", "limited", "requirement"]):
            insights.append("This problem has constraints - consider constraint satisfaction techniques")
            abstractions.append("constraint_problem")
        
        # Generate meta-insights
        if len(abstractions) > 1:
            insights.append(f"This is a multi-faceted problem involving {', '.join(abstractions)}")
        
        return {
            "move_name": "abstraction",
            "input_state": {"problem_elements": problem_elements},
            "output_state": {"abstractions": abstractions},
            "cognitive_level": CognitiveLevel.METACOGNITIVE,
            "rationale": "Abstracting problem to identify higher-level patterns",
            "confidence": 0.7,
            "insights": insights,
            "success": len(abstractions) > 0,
        }
    
    async def _decomposition_move(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Decompose the problem into smaller, manageable subproblems."""
        
        insights = []
        subproblems = []
        
        # Decompose by objectives
        for i, objective in enumerate(problem.objectives):
            subproblem = {
                "id": f"obj_{i}",
                "description": f"Achieve objective: {objective}",
                "type": "objective_subproblem",
            }
            subproblems.append(subproblem)
            insights.append(f"Subproblem {i+1}: Focus on achieving '{objective}'")
        
        # Decompose by constraints
        for i, constraint in enumerate(problem.constraints):
            subproblem = {
                "id": f"con_{i}",
                "description": f"Satisfy constraint: {constraint}",
                "type": "constraint_subproblem",
            }
            subproblems.append(subproblem)
            insights.append(f"Constraint handling: Ensure '{constraint}' is satisfied")
        
        # Decompose by problem phases
        if problem.problem_type == "planning":
            phases = ["analysis", "design", "implementation", "evaluation"]
        elif problem.problem_type == "optimization":
            phases = ["modeling", "solving", "validation", "refinement"]
        else:
            phases = ["understanding", "solution_generation", "evaluation", "implementation"]
        
        for i, phase in enumerate(phases):
            subproblem = {
                "id": f"phase_{i}",
                "description": f"{phase.title()} phase of problem solving",
                "type": "phase_subproblem",
            }
            subproblems.append(subproblem)
            insights.append(f"Phase {i+1}: {phase.title()} - focus on {phase}-specific activities")
        
        # Identify dependencies
        dependencies = []
        if len(subproblems) > 1:
            # Simple heuristic: phases depend on previous phases
            phase_problems = [sp for sp in subproblems if sp["type"] == "phase_subproblem"]
            for i in range(1, len(phase_problems)):
                dependencies.append({
                    "dependent": phase_problems[i]["id"],
                    "prerequisite": phase_problems[i-1]["id"],
                })
        
        insights.append(f"Identified {len(subproblems)} subproblems with {len(dependencies)} dependencies")
        
        return {
            "move_name": "decomposition",
            "input_state": {"original_problem": problem.title},
            "output_state": {"subproblems": subproblems, "dependencies": dependencies},
            "cognitive_level": CognitiveLevel.EXECUTIVE,
            "rationale": "Breaking down complex problem into manageable subproblems",
            "confidence": 0.8,
            "insights": insights,
            "success": len(subproblems) > 0,
        }
    
    async def _perspective_shift(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Shift perspective to view the problem from different angles."""
        
        insights = []
        perspectives = []
        
        # Stakeholder perspectives
        stakeholder_perspectives = {
            "user": "How does this problem affect end users?",
            "developer": "What technical challenges does this present?",
            "manager": "What are the resource and timeline implications?",
            "customer": "What value does solving this create?",
            "competitor": "How might competitors approach this?",
        }
        
        for stakeholder, question in stakeholder_perspectives.items():
            perspectives.append({
                "type": "stakeholder",
                "name": stakeholder,
                "question": question,
            })
            insights.append(f"From {stakeholder} perspective: {question}")
        
        # Temporal perspectives
        temporal_perspectives = [
            "What would this problem look like 10 years ago?",
            "How might this problem evolve in the future?",
            "What are the short-term vs long-term implications?",
        ]
        
        for question in temporal_perspectives:
            perspectives.append({
                "type": "temporal",
                "question": question,
            })
            insights.append(f"Temporal view: {question}")
        
        # Scale perspectives
        scale_perspectives = [
            "What if this problem were 10x larger in scope?",
            "What if this problem were much smaller?",
            "What are the global vs local implications?",
        ]
        
        for question in scale_perspectives:
            perspectives.append({
                "type": "scale",
                "question": question,
            })
            insights.append(f"Scale consideration: {question}")
        
        # Inversion perspective
        inversion_insight = f"What if we tried to make the problem worse instead of better?"
        perspectives.append({
            "type": "inversion",
            "question": inversion_insight,
        })
        insights.append(f"Inversion thinking: {inversion_insight}")
        
        return {
            "move_name": "perspective_shift",
            "input_state": {"original_perspective": "default"},
            "output_state": {"perspectives": perspectives},
            "cognitive_level": CognitiveLevel.METACOGNITIVE,
            "rationale": "Viewing problem from multiple perspectives",
            "confidence": 0.6,
            "insights": insights,
            "success": len(perspectives) > 0,
        }
    
    async def _constraint_relaxation(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Relax constraints to explore alternative solution spaces."""
        
        insights = []
        relaxations = []
        
        # Relax explicit constraints
        for i, constraint in enumerate(problem.constraints):
            relaxation = {
                "original": constraint,
                "relaxed": f"Partially satisfy: {constraint}",
                "type": "partial_relaxation",
            }
            relaxations.append(relaxation)
            insights.append(f"What if we only partially satisfied: {constraint}?")
            
            # Complete removal
            relaxation = {
                "original": constraint,
                "relaxed": f"Remove constraint: {constraint}",
                "type": "complete_removal",
            }
            relaxations.append(relaxation)
            insights.append(f"What if we completely removed: {constraint}?")
        
        # Relax implicit constraints
        implicit_constraints = [
            "Must use existing technology",
            "Must work within current budget",
            "Must be implemented immediately",
            "Must satisfy all stakeholders equally",
            "Must be perfect solution",
        ]
        
        for constraint in implicit_constraints:
            relaxation = {
                "original": constraint,
                "relaxed": f"Negotiate: {constraint}",
                "type": "negotiation",
            }
            relaxations.append(relaxation)
            insights.append(f"What if we negotiated: {constraint}?")
        
        # Time constraint relaxation
        insights.append("What if we had unlimited time to solve this?")
        insights.append("What if we had to solve this in half the time?")
        
        # Resource constraint relaxation
        insights.append("What if we had unlimited resources?")
        insights.append("What if we had to solve this with minimal resources?")
        
        return {
            "move_name": "constraint_relaxation",
            "input_state": {"original_constraints": problem.constraints},
            "output_state": {"relaxations": relaxations},
            "cognitive_level": CognitiveLevel.EXECUTIVE,
            "rationale": "Exploring solutions by relaxing constraints",
            "confidence": 0.5,
            "insights": insights,
            "success": len(relaxations) > 0,
        }
    
    async def _resource_substitution(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Explore alternative resources and approaches."""
        
        insights = []
        substitutions = []
        
        # Technology substitutions
        tech_substitutions = {
            "manual process": "automated system",
            "expensive solution": "cost-effective alternative",
            "complex system": "simple approach",
            "new technology": "proven technology",
            "centralized approach": "distributed approach",
        }
        
        for original, alternative in tech_substitutions.items():
            substitutions.append({
                "category": "technology",
                "original": original,
                "alternative": alternative,
            })
            insights.append(f"Instead of {original}, consider {alternative}")
        
        # Resource substitutions
        resource_substitutions = {
            "human resources": "automation or AI",
            "financial resources": "partnerships or crowdfunding",
            "time resources": "parallel processing or outsourcing",
            "physical space": "virtual or cloud-based solutions",
            "expertise": "training, consulting, or collaboration",
        }
        
        for original, alternative in resource_substitutions.items():
            substitutions.append({
                "category": "resources",
                "original": original,
                "alternative": alternative,
            })
            insights.append(f"Instead of relying on {original}, consider {alternative}")
        
        # Approach substitutions
        approach_substitutions = {
            "top-down approach": "bottom-up approach",
            "sequential process": "parallel process",
            "individual effort": "collaborative effort",
            "internal solution": "external partnership",
            "custom development": "existing tools/platforms",
        }
        
        for original, alternative in approach_substitutions.items():
            substitutions.append({
                "category": "approach",
                "original": original,
                "alternative": alternative,
            })
            insights.append(f"Instead of {original}, try {alternative}")
        
        return {
            "move_name": "resource_substitution",
            "input_state": {"current_resources": "standard"},
            "output_state": {"substitutions": substitutions},
            "cognitive_level": CognitiveLevel.OPERATIONAL,
            "rationale": "Exploring alternative resources and approaches",
            "confidence": 0.6,
            "insights": insights,
            "success": len(substitutions) > 0,
        }
    
    async def _temporal_reasoning(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Apply temporal reasoning to understand timing and sequencing."""
        
        insights = []
        temporal_aspects = []
        
        # Timeline analysis
        timeline_points = [
            "Problem emergence: When did this problem first appear?",
            "Current state: What is the situation right now?",
            "Near future: What happens if we don't act soon?",
            "Long-term: What are the long-term implications?",
        ]
        
        for point in timeline_points:
            temporal_aspects.append({
                "type": "timeline",
                "description": point,
            })
            insights.append(point)
        
        # Sequencing considerations
        sequencing_insights = [
            "What must happen first before other actions can be taken?",
            "What can be done in parallel to save time?",
            "What are the critical path dependencies?",
            "Where are the potential bottlenecks in timing?",
        ]
        
        for insight in sequencing_insights:
            temporal_aspects.append({
                "type": "sequencing",
                "description": insight,
            })
            insights.append(insight)
        
        # Temporal constraints
        temporal_constraints = [
            "Deadlines: What are the hard time limits?",
            "Windows of opportunity: When is the best time to act?",
            "Seasonal factors: Are there time-dependent variables?",
            "Synchronization: What needs to happen simultaneously?",
        ]
        
        for constraint in temporal_constraints:
            temporal_aspects.append({
                "type": "constraint",
                "description": constraint,
            })
            insights.append(constraint)
        
        # Urgency vs importance analysis
        urgency_insights = [
            "Is this problem urgent, important, both, or neither?",
            "What happens if we delay action by a week? A month? A year?",
            "Are there quick wins that can provide immediate value?",
        ]
        
        for insight in urgency_insights:
            temporal_aspects.append({
                "type": "urgency",
                "description": insight,
            })
            insights.append(insight)
        
        return {
            "move_name": "temporal_reasoning",
            "input_state": {"time_perspective": "static"},
            "output_state": {"temporal_aspects": temporal_aspects},
            "cognitive_level": CognitiveLevel.EXECUTIVE,
            "rationale": "Analyzing temporal dimensions of the problem",
            "confidence": 0.7,
            "insights": insights,
            "success": len(temporal_aspects) > 0,
        }
    
    async def _causal_analysis(
        self,
        problem: Problem,
        cognitive_state: CognitiveState,
        step_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Analyze causal relationships and root causes."""
        
        insights = []
        causal_factors = []
        
        # Root cause analysis
        potential_causes = [
            "Technical factors: System limitations, bugs, or design flaws",
            "Human factors: Skills gaps, communication issues, or resistance to change",
            "Process factors: Inefficient workflows, unclear procedures, or bottlenecks",
            "Environmental factors: Market conditions, regulations, or external pressures",
            "Resource factors: Insufficient funding, time, or personnel",
        ]
        
        for cause in potential_causes:
            causal_factors.append({
                "type": "root_cause",
                "description": cause,
            })
            insights.append(f"Consider: {cause}")
        
        # Effect analysis
        potential_effects = [
            "Immediate effects: What happens right away if this isn't solved?",
            "Ripple effects: What secondary problems might emerge?",
            "Long-term consequences: What are the ultimate outcomes?",
            "Stakeholder impacts: Who is affected and how?",
        ]
        
        for effect in potential_effects:
            causal_factors.append({
                "type": "effect",
                "description": effect,
            })
            insights.append(effect)
        
        # Causal loops
        loop_insights = [
            "Are there vicious cycles that make the problem worse over time?",
            "Are there virtuous cycles we could leverage for solutions?",
            "What feedback loops exist in this system?",
        ]
        
        for insight in loop_insights:
            causal_factors.append({
                "type": "feedback_loop",
                "description": insight,
            })
            insights.append(insight)
        
        # Leverage points
        leverage_insights = [
            "Where could small changes create big impacts?",
            "What are the high-leverage intervention points?",
            "Which causes, if addressed, would solve multiple problems?",
        ]
        
        for insight in leverage_insights:
            causal_factors.append({
                "type": "leverage",
                "description": insight,
            })
            insights.append(insight)
        
        return {
            "move_name": "causal_analysis",
            "input_state": {"analysis_depth": "surface"},
            "output_state": {"causal_factors": causal_factors},
            "cognitive_level": CognitiveLevel.OPERATIONAL,
            "rationale": "Analyzing causal relationships and root causes",
            "confidence": 0.7,
            "insights": insights,
            "success": len(causal_factors) > 0,
        }
    
    async def _synthesize_solution(self, problem: Problem, insights: List[str]) -> Optional[Solution]:
        """Synthesize a solution from accumulated insights."""
        
        if not insights:
            return None
        
        # Organize insights by categories
        insight_categories = {
            "analogies": [i for i in insights if "similar" in i.lower() or "analogy" in i.lower()],
            "decomposition": [i for i in insights if "subproblem" in i.lower() or "phase" in i.lower()],
            "perspectives": [i for i in insights if "perspective" in i.lower() or "view" in i.lower()],
            "constraints": [i for i in insights if "constraint" in i.lower() or "relax" in i.lower()],
            "resources": [i for i in insights if "resource" in i.lower() or "instead of" in i.lower()],
            "temporal": [i for i in insights if "time" in i.lower() or "when" in i.lower()],
            "causal": [i for i in insights if "cause" in i.lower() or "effect" in i.lower()],
        }
        
        solution_text = f"Cognitive moves analysis of '{problem.title}':\n\n"
        
        # Add insights by category
        for category, category_insights in insight_categories.items():
            if category_insights:
                solution_text += f"{category.title()} Insights:\n"
                for insight in category_insights[:3]:  # Top 3 per category
                    solution_text += f"• {insight}\n"
                solution_text += "\n"
        
        # Add synthesis
        solution_text += "Synthesis:\n"
        solution_text += f"Applied {len(self.move_history)} cognitive moves to analyze this problem. "
        
        successful_moves = [move for move in self.move_history if move["success"]]
        if successful_moves:
            solution_text += f"Most effective moves were: {', '.join([move['move'] for move in successful_moves[:3]])}. "
        
        solution_text += f"Generated {len(insights)} insights across multiple dimensions of analysis."
        
        # Calculate confidence based on insight diversity and move success
        insight_diversity = len([cat for cat, insights in insight_categories.items() if insights])
        move_success_rate = len(successful_moves) / max(1, len(self.move_history))
        
        confidence = min(0.7, (insight_diversity / 7.0) * 0.5 + move_success_rate * 0.5)
        
        solution = Solution(
            problem_id=problem.id,
            solution_text=solution_text,
            confidence=confidence,
            completeness=0.6,  # Cognitive moves provide broad analysis
            feasibility=0.5,   # Insights need further development
            solving_time=None,
            reasoning_steps=len(self.move_history),
        )
        
        return solution
