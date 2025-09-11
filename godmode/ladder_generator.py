"""
Progressive ladder generation for GODMODE - builds prior and future question ladders
"""

import asyncio
import hashlib
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .schemas import QuestionNode, ScenarioLane, Trigger, CrossLink
from .models import ModelRouter, calculate_expected_info_gain, calculate_coherence, calculate_effort_penalty


class CognitiveMove(Enum):
    """Cognitive moves in progressive order"""
    DEFINE = "define"
    SCOPE = "scope" 
    QUANTIFY = "quantify"
    COMPARE = "compare"
    SIMULATE = "simulate"
    DECIDE = "decide"
    COMMIT = "commit"


@dataclass
class LadderConfig:
    """Configuration for ladder generation"""
    beam_width: int = 4
    depth_back: int = 4
    depth_forward: int = 5
    prune_threshold: float = 0.18
    max_lanes: int = 5
    min_lanes: int = 3


class LadderGenerator:
    """Generates progressive question ladders using enumerate/rerank/stitch pipeline"""
    
    def __init__(self, model_router: ModelRouter, config: LadderConfig = None):
        self.router = model_router
        self.config = config or LadderConfig()
        self.question_counter = 0
        self.lane_counter = 0
        self.seen_questions: Set[str] = set()  # For deduplication
    
    def _get_next_question_id(self) -> str:
        """Generate next question ID"""
        self.question_counter += 1
        return f"Q{self.question_counter:03d}"
    
    def _get_next_lane_id(self) -> str:
        """Generate next lane ID"""
        self.lane_counter += 1
        lane_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"S-{lane_letters[self.lane_counter - 1]}"
    
    def _hash_question(self, text: str) -> str:
        """Create canonical hash for question deduplication"""
        # Normalize text for hashing
        normalized = " ".join(text.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    
    def _is_duplicate(self, question_text: str) -> bool:
        """Check if question is a near-duplicate of existing ones"""
        question_hash = self._hash_question(question_text)
        if question_hash in self.seen_questions:
            return True
        self.seen_questions.add(question_hash)
        return False
    
    async def generate_priors(self, current_question: str, context: str = "") -> List[QuestionNode]:
        """Generate backward reasoning ladder (priors)"""
        priors = []
        
        # Phase 1: ENUMERATE - Generate premise mining prompts
        premise_prompt = self._build_premise_mining_prompt(current_question, context)
        candidates = await self.router.enumerate(premise_prompt, self.config.beam_width * 2)
        
        # Filter and parse candidates
        parsed_candidates = []
        for candidate in candidates:
            questions = self._parse_question_candidates(candidate)
            parsed_candidates.extend(questions)
        
        # Remove duplicates
        unique_candidates = []
        for q_text in parsed_candidates:
            if not self._is_duplicate(q_text) and q_text.strip():
                unique_candidates.append(q_text)
        
        if not unique_candidates:
            return []
        
        # Phase 2: RERANK - Score by expected info gain
        rerank_query = f"Questions that would make this trivial to answer: {current_question}"
        ranked_candidates = await self.router.rerank(rerank_query, unique_candidates, self.config.beam_width)
        
        # Phase 3: STITCH - Build progressive ladder
        priors = await self._build_prior_ladder(ranked_candidates, current_question, context)
        
        return priors
    
    async def generate_scenarios(self, current_question: str, context: str = "", 
                               priors: List[QuestionNode] = None) -> List[ScenarioLane]:
        """Generate forward reasoning scenarios (future ladders)"""
        scenarios = []
        
        # Phase 1: Generate lane theses
        lane_prompt = self._build_scenario_prompt(current_question, context, priors)
        lane_candidates = await self.router.enumerate(lane_prompt, self.config.max_lanes * 2)
        
        # Parse lane theses
        lane_theses = self._parse_lane_theses(lane_candidates)
        
        # Phase 2: Rerank lane theses
        rerank_query = f"Alternative future paths after answering: {current_question}"
        ranked_theses = await self.router.rerank(rerank_query, lane_theses, self.config.max_lanes)
        
        # Phase 3: Build scenario lanes
        for i, (thesis, score) in enumerate(ranked_theses[:self.config.max_lanes]):
            if score > 0.3:  # Minimum relevance threshold
                lane = await self._build_scenario_lane(thesis, current_question, context, i)
                if lane:
                    scenarios.append(lane)
        
        # Ensure minimum number of lanes
        while len(scenarios) < self.config.min_lanes and len(scenarios) < len(ranked_theses):
            thesis = ranked_theses[len(scenarios)][0]
            lane = await self._build_scenario_lane(thesis, current_question, context, len(scenarios))
            if lane:
                scenarios.append(lane)
        
        return scenarios
    
    def _build_premise_mining_prompt(self, current_question: str, context: str) -> str:
        """Build prompt for premise mining (backward reasoning)"""
        return f"""
You are a premise mining expert. Given this question, identify the hidden assumptions and prerequisites that must be resolved first.

CURRENT QUESTION: {current_question}
CONTEXT: {context}

Generate 4-6 foundational questions that would make the current question trivial to answer if resolved first. Focus on:
1. DEFINITIONS: What key terms need clarification?
2. SCOPE: What boundaries and constraints apply?
3. QUANTIFICATION: What metrics or measurements are needed?
4. COMPARISON: What alternatives or benchmarks matter?

Format as a simple list:
- Question 1
- Question 2
- Question 3
...

Each question should add a new constraint, metric, frame, stake, or counterfactual.
"""
    
    def _build_scenario_prompt(self, current_question: str, context: str, 
                             priors: List[QuestionNode] = None) -> str:
        """Build prompt for scenario generation"""
        prior_context = ""
        if priors:
            prior_context = "\nPRIOR QUESTIONS RESOLVED:\n" + "\n".join([f"- {p.text}" for p in priors[:3]])
        
        return f"""
You are a scenario planning expert. Generate 3-5 alternative future paths after answering this question.

CURRENT QUESTION: {current_question}
CONTEXT: {context}{prior_context}

Generate distinct scenario lanes representing different strategic directions or outcomes. Each lane should have:
- A clear thesis/theme (e.g., "Growth-Focused", "Risk-Minimized", "Innovation-Led")
- A one-line description of the strategic direction

Format as:
Lane A: [Name] - [One-line thesis]
Lane B: [Name] - [One-line thesis]
...

Examples:
Lane A: Career-Max - Optimize for maximum career advancement and compensation
Lane B: Lifestyle-Opt - Balance career growth with personal fulfillment and flexibility
Lane C: Option-Value - Preserve maximum future optionality and learning opportunities
"""
    
    def _parse_question_candidates(self, candidate_text: str) -> List[str]:
        """Parse question candidates from generated text"""
        questions = []
        lines = candidate_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Remove bullet points, numbers, etc.
            line = line.lstrip('- •*1234567890. ')
            
            # Must be a question or question-like statement
            if line and (line.endswith('?') or any(word in line.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which'])):
                questions.append(line)
        
        return questions
    
    def _parse_lane_theses(self, candidates: List[str]) -> List[str]:
        """Parse lane theses from generated text"""
        theses = []
        
        for candidate in candidates:
            lines = candidate.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Look for "Lane X: Name - Description" format
                if 'Lane' in line and ':' in line and '-' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        thesis = parts[1].strip()
                        if thesis:
                            theses.append(thesis)
                elif line and not line.startswith(('Lane', 'Generate', 'Examples')):
                    # Fallback: treat as thesis if it looks substantive
                    if len(line.split()) >= 3:
                        theses.append(line)
        
        return theses
    
    async def _build_prior_ladder(self, ranked_candidates: List[Tuple[str, float]], 
                                current_question: str, context: str) -> List[QuestionNode]:
        """Build progressive prior ladder with cognitive moves"""
        if not ranked_candidates:
            return []
        
        priors = []
        cognitive_moves = [CognitiveMove.DEFINE, CognitiveMove.SCOPE, CognitiveMove.QUANTIFY, CognitiveMove.COMPARE]
        
        for i, (question_text, relevance_score) in enumerate(ranked_candidates[:self.config.depth_back]):
            if i >= len(cognitive_moves):
                move = CognitiveMove.SIMULATE
            else:
                move = cognitive_moves[i]
            
            # Generate delta nuance
            delta_nuance = await self._generate_delta_nuance(question_text, priors, move.value)
            
            # Calculate metrics
            info_gain = calculate_expected_info_gain(question_text, context)
            coherence = calculate_coherence(question_text, [p.text for p in priors])
            effort = calculate_effort_penalty(question_text)
            
            # Confidence based on relevance and coherence
            confidence = min(relevance_score * coherence, 1.0)
            
            # Build dependencies
            builds_on = []
            if priors:
                # Build on previous question in sequence
                builds_on = [priors[-1].id]
            
            prior = QuestionNode(
                id=self._get_next_question_id(),
                text=question_text,
                level=i + 1,
                cognitive_move=move.value,
                builds_on=builds_on,
                delta_nuance=delta_nuance,
                expected_info_gain=info_gain,
                confidence=confidence,
                tags=self._extract_tags(question_text)
            )
            
            priors.append(prior)
            
            # Prune if info gain too low for consecutive levels
            if len(priors) >= 2 and priors[-1].expected_info_gain < self.config.prune_threshold and priors[-2].expected_info_gain < self.config.prune_threshold:
                break
        
        return priors
    
    async def _build_scenario_lane(self, thesis: str, current_question: str, 
                                 context: str, lane_index: int) -> Optional[ScenarioLane]:
        """Build a single scenario lane with progressive questions"""
        # Parse thesis into name and description
        if ' - ' in thesis:
            name, description = thesis.split(' - ', 1)
        else:
            name = f"Scenario {chr(65 + lane_index)}"  # A, B, C...
            description = thesis
        
        lane_id = self._get_next_lane_id()
        
        # Generate questions for this lane
        lane_prompt = f"""
Generate a progressive sequence of 3-5 questions that explore this scenario after answering: {current_question}

SCENARIO: {name} - {description}
CONTEXT: {context}

Build a logical progression using these cognitive moves in order:
1. DEFINE: What does this scenario specifically entail?
2. SCOPE: What are the boundaries and constraints?
3. QUANTIFY: What metrics or measurements matter?
4. COMPARE: What are the trade-offs vs alternatives?
5. SIMULATE: How would this play out in practice?

Format as numbered list:
1. [Question text]
2. [Question text]
...
"""
        
        candidates = await self.router.enumerate(lane_prompt, self.config.beam_width)
        
        # Parse and build lane questions
        lane_questions = []
        cognitive_moves = [CognitiveMove.DEFINE, CognitiveMove.SCOPE, CognitiveMove.QUANTIFY, 
                         CognitiveMove.COMPARE, CognitiveMove.SIMULATE, CognitiveMove.DECIDE]
        
        all_questions = []
        for candidate in candidates:
            questions = self._parse_question_candidates(candidate)
            all_questions.extend(questions)
        
        # Remove duplicates and select best
        unique_questions = []
        for q in all_questions:
            if not self._is_duplicate(q) and q.strip():
                unique_questions.append(q)
        
        # Rerank for this lane
        rerank_query = f"Questions that explore the scenario: {description}"
        if unique_questions:
            ranked_questions = await self.router.rerank(rerank_query, unique_questions, self.config.depth_forward)
        else:
            return None
        
        # Build progressive questions
        for i, (question_text, score) in enumerate(ranked_questions[:self.config.depth_forward]):
            if i >= len(cognitive_moves):
                move = CognitiveMove.COMMIT
            else:
                move = cognitive_moves[i]
            
            # Generate delta nuance
            delta_nuance = await self._generate_delta_nuance(question_text, lane_questions, move.value, description)
            
            # Calculate metrics
            info_gain = calculate_expected_info_gain(question_text, context)
            coherence = calculate_coherence(question_text, [q.text for q in lane_questions], description)
            
            confidence = min(score * coherence, 1.0)
            
            # Build dependencies within lane
            builds_on = []
            if lane_questions:
                builds_on = [lane_questions[-1].id]
            
            # Generate triggers
            triggers = self._generate_triggers(question_text, move.value)
            
            # Check for natural end
            natural_end = (move == CognitiveMove.COMMIT or 
                          (i >= 2 and info_gain < self.config.prune_threshold))
            
            question = QuestionNode(
                id=self._get_next_question_id(),
                text=question_text,
                level=i + 1,
                cognitive_move=move.value,
                builds_on=builds_on,
                delta_nuance=delta_nuance,
                expected_info_gain=info_gain,
                confidence=confidence,
                triggers=triggers,
                natural_end=natural_end,
                tags=self._extract_tags(question_text) + [f"scenario_{lane_index}"]
            )
            
            lane_questions.append(question)
            
            if natural_end:
                break
        
        if not lane_questions:
            return None
        
        return ScenarioLane(
            id=lane_id,
            name=name.strip(),
            description=description.strip(),
            lane=lane_questions,
            cross_links=[]  # TODO: Implement cross-links between lanes
        )
    
    async def _generate_delta_nuance(self, question_text: str, previous_questions: List[QuestionNode], 
                                   cognitive_move: str, lane_context: str = "") -> str:
        """Generate delta nuance for a question"""
        # Simple heuristic-based approach - could be enhanced with LLM
        move_nuances = {
            "define": "Clarifies key definitions and terminology",
            "scope": "Establishes boundaries and constraints", 
            "quantify": "Adds measurable metrics and thresholds",
            "compare": "Introduces comparative analysis and trade-offs",
            "simulate": "Explores implementation scenarios and outcomes",
            "decide": "Focuses on decision criteria and selection",
            "commit": "Addresses execution and commitment mechanisms"
        }
        
        base_nuance = move_nuances.get(cognitive_move, "Adds new analytical dimension")
        
        # Customize based on question content
        if "cost" in question_text.lower() or "$" in question_text:
            return f"{base_nuance} with financial considerations"
        elif "time" in question_text.lower() or "when" in question_text.lower():
            return f"{base_nuance} with temporal constraints"
        elif "risk" in question_text.lower():
            return f"{base_nuance} with risk assessment"
        elif "stakeholder" in question_text.lower() or "team" in question_text.lower():
            return f"{base_nuance} with stakeholder perspectives"
        
        return base_nuance
    
    def _generate_triggers(self, question_text: str, cognitive_move: str) -> List[Trigger]:
        """Generate triggers for when this question becomes relevant"""
        triggers = []
        
        # Time-based triggers
        if any(word in question_text.lower() for word in ["quarterly", "monthly", "annual", "deadline"]):
            triggers.append(Trigger(type="time", detail="Periodic review cycle"))
        
        # Metric-based triggers
        if any(word in question_text.lower() for word in ["threshold", "target", "goal", "kpi"]):
            triggers.append(Trigger(type="metric", detail="Performance threshold reached"))
        
        # Event-based triggers
        if any(word in question_text.lower() for word in ["launch", "release", "implement", "deploy"]):
            triggers.append(Trigger(type="event", detail="Implementation milestone"))
        
        # Answer-change triggers for decision points
        if cognitive_move in ["decide", "commit"]:
            triggers.append(Trigger(type="answer_change", detail="Previous assumptions invalidated"))
        
        return triggers
    
    def _extract_tags(self, question_text: str) -> List[str]:
        """Extract domain and intent tags from question text"""
        tags = []
        
        # Domain tags
        domain_keywords = {
            "business": ["revenue", "profit", "market", "customer", "business", "strategy"],
            "technical": ["system", "architecture", "code", "technical", "implementation", "api"],
            "financial": ["cost", "budget", "investment", "roi", "financial", "price"],
            "operational": ["process", "workflow", "operation", "efficiency", "resource"],
            "legal": ["compliance", "regulation", "legal", "contract", "policy"],
            "hr": ["team", "hiring", "culture", "employee", "talent", "management"]
        }
        
        question_lower = question_text.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                tags.append(domain)
        
        # Intent tags
        if "?" in question_text:
            if question_text.lower().startswith(("what", "how", "why")):
                tags.append("exploratory")
            elif question_text.lower().startswith(("should", "would", "could")):
                tags.append("decisional")
            elif question_text.lower().startswith(("when", "where")):
                tags.append("logistical")
        
        return tags[:3]  # Limit to 3 most relevant tags