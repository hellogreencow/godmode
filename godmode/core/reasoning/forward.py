"""Forward reasoning for generating FUTURE scenario lanes."""

import asyncio
import random
from typing import List, Optional, Dict, Any, Tuple
from ...models.core import Question, Lane, CognitiveMove, Trigger, TriggerType, CrossLink
from ...models.commands import Budgets
from .cognitive_moves import CognitiveMoveProgression
from .scoring import ScoreCalculator


class ForwardReasoning:
    """
    Generates FUTURE ladders through forward reasoning.
    
    Creates 3-5 scenario lanes with alternative futures, each with
    variable branch depths and natural endings.
    """
    
    def __init__(self):
        self.move_progression = CognitiveMoveProgression()
        self.scorer = ScoreCalculator()
        self._question_counter = 0
        
        # Lane templates for different scenario types
        self.lane_templates = [
            {
                "name": "Direct Path",
                "description": "Most straightforward approach to the goal",
                "focus": "efficiency",
                "cognitive_emphasis": [CognitiveMove.DEFINE, CognitiveMove.QUANTIFY, CognitiveMove.DECIDE]
            },
            {
                "name": "Exploratory",
                "description": "Thorough exploration of alternatives",
                "focus": "comprehensiveness", 
                "cognitive_emphasis": [CognitiveMove.SCOPE, CognitiveMove.COMPARE, CognitiveMove.SIMULATE]
            },
            {
                "name": "Risk-Aware",
                "description": "Cautious approach considering potential downsides",
                "focus": "risk_management",
                "cognitive_emphasis": [CognitiveMove.SCOPE, CognitiveMove.SIMULATE, CognitiveMove.COMPARE]
            },
            {
                "name": "Innovative",
                "description": "Creative alternatives and novel approaches",
                "focus": "innovation",
                "cognitive_emphasis": [CognitiveMove.DEFINE, CognitiveMove.SIMULATE, CognitiveMove.DECIDE]
            },
            {
                "name": "Resource-Optimized",
                "description": "Minimizing resource requirements and constraints",
                "focus": "efficiency",
                "cognitive_emphasis": [CognitiveMove.QUANTIFY, CognitiveMove.COMPARE, CognitiveMove.DECIDE]
            }
        ]
    
    async def enumerate_futures(
        self, 
        current_question: str,
        context: Optional[str], 
        budgets: Budgets
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: ENUMERATE - Generate diverse future scenario candidates.
        
        Proposes 3-5 lane theses and generates initial questions for each.
        """
        # Generate scenario lane proposals
        lane_count = min(5, max(3, budgets.beam_width))
        selected_templates = random.sample(self.lane_templates, lane_count)
        
        # Generate futures for each lane
        future_tasks = []
        for i, template in enumerate(selected_templates):
            lane_id = f"S-{chr(65 + i)}"  # S-A, S-B, S-C, etc.
            task = self._generate_lane_candidates(
                lane_id, template, current_question, context, budgets
            )
            future_tasks.append(task)
        
        lane_candidates = await asyncio.gather(*future_tasks)
        
        return lane_candidates
    
    async def rerank_candidates(self, lane_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phase 2: RERANK - Score lane candidates by overall expected value.
        """
        scored_lanes = []
        
        for lane_data in lane_candidates:
            questions = lane_data["questions"]
            
            # Calculate aggregate scores for the lane
            total_info_gain = sum(q.expected_info_gain for q in questions) / len(questions) if questions else 0
            avg_confidence = sum(q.confidence for q in questions) / len(questions) if questions else 0
            
            # Lane diversity bonus (different cognitive focuses)
            cognitive_diversity = len(set(q.cognitive_move for q in questions)) / len(CognitiveMove) if questions else 0
            
            # Natural progression bonus
            progression_score = self._calculate_progression_quality(questions)
            
            lane_score = (
                0.4 * total_info_gain + 
                0.3 * avg_confidence + 
                0.2 * cognitive_diversity + 
                0.1 * progression_score
            )
            
            lane_data["lane_score"] = lane_score
            scored_lanes.append(lane_data)
        
        # Sort by lane score
        scored_lanes.sort(key=lambda x: x["lane_score"], reverse=True)
        
        return scored_lanes
    
    async def stitch_scenarios(self, ranked_lane_candidates: List[Dict[str, Any]]) -> List[Lane]:
        """
        Phase 3: STITCH - Create final Lane objects with proper relationships.
        """
        lanes = []
        
        for lane_data in ranked_lane_candidates:
            # Build questions with proper builds_on relationships
            stitched_questions = await self._stitch_lane_questions(lane_data["questions"])
            
            # Add triggers to appropriate questions
            self._add_triggers_to_questions(stitched_questions)
            
            # Detect natural endings
            self._mark_natural_endings(stitched_questions)
            
            # Create Lane object
            lane = Lane(
                id=lane_data["id"],
                name=lane_data["name"],
                description=lane_data["description"],
                lane=stitched_questions,
                cross_links=[]  # Will be added later if needed
            )
            
            lanes.append(lane)
        
        # Add cross-links between lanes
        self._add_cross_links(lanes)
        
        return lanes
    
    async def expand_around_node(
        self, 
        node: Question, 
        user_answer: Optional[str],
        budgets: Budgets
    ) -> List[Question]:
        """Expand questions around a specific node."""
        expanded_questions = []
        
        # Generate follow-up questions based on the node's cognitive move
        next_moves = self.move_progression.get_next_moves(node.cognitive_move)
        
        for next_move in list(next_moves)[:budgets.beam_width]:
            follow_up = await self._generate_follow_up_question(
                node, next_move, user_answer
            )
            if follow_up:
                expanded_questions.append(follow_up)
        
        return expanded_questions
    
    async def continue_from_node(
        self, 
        node: Question,
        budgets: Budgets
    ) -> List[Question]:
        """Continue a question sequence from a specific node."""
        continued_questions = []
        
        # Generate next logical questions
        for i in range(min(3, budgets.depth_fwd)):
            if not continued_questions:
                # First continuation from the node
                next_question = await self._generate_continuation_question(
                    node, node.level + 1
                )
            else:
                # Continue from last generated question
                next_question = await self._generate_continuation_question(
                    continued_questions[-1], continued_questions[-1].level + 1
                )
            
            if next_question:
                continued_questions.append(next_question)
            else:
                break  # No more logical continuations
        
        return continued_questions
    
    async def _generate_lane_candidates(
        self,
        lane_id: str,
        template: Dict[str, Any],
        current_question: str,
        context: Optional[str],
        budgets: Budgets
    ) -> Dict[str, Any]:
        """Generate question candidates for a specific lane."""
        questions = []
        
        # Generate questions following the template's cognitive emphasis
        for level in range(1, budgets.depth_fwd + 1):
            # Choose cognitive move based on template emphasis and level
            cognitive_move = self._choose_cognitive_move_for_level(
                level, template["cognitive_emphasis"]
            )
            
            # Generate question for this level
            question_text = self._generate_question_text(
                current_question, template, cognitive_move, level, context
            )
            
            question = Question(
                id=f"Q{lane_id[-1]}{level}",  # QA1, QB2, etc.
                text=question_text,
                level=level,
                cognitive_move=cognitive_move,
                builds_on=[],  # Will be filled in stitching phase
                delta_nuance=self._generate_delta_nuance(cognitive_move, template),
                expected_info_gain=0.0,  # Will be calculated
                confidence=0.0,  # Will be calculated
                tags=[template["focus"], cognitive_move.value]
            )
            
            # Calculate scores
            question.expected_info_gain = self.scorer.calculate_info_gain(question)
            question.confidence = self.scorer.calculate_confidence(question)
            
            questions.append(question)
            
            # Early termination if info gain is too low
            if level > 2 and question.expected_info_gain < budgets.prune_if_info_gain_below:
                break
        
        return {
            "id": lane_id,
            "name": template["name"],
            "description": template["description"],
            "questions": questions,
            "template": template
        }
    
    def _choose_cognitive_move_for_level(
        self, 
        level: int, 
        emphasis_moves: List[CognitiveMove]
    ) -> CognitiveMove:
        """Choose appropriate cognitive move for a given level and emphasis."""
        # Map levels to default moves
        level_defaults = {
            1: CognitiveMove.DEFINE,
            2: CognitiveMove.SCOPE,
            3: CognitiveMove.QUANTIFY,
            4: CognitiveMove.COMPARE,
            5: CognitiveMove.SIMULATE,
            6: CognitiveMove.DECIDE,
            7: CognitiveMove.COMMIT
        }
        
        default_move = level_defaults.get(level, CognitiveMove.DECIDE)
        
        # Bias towards emphasis moves
        if emphasis_moves and random.random() < 0.7:  # 70% chance to use emphasis
            return random.choice(emphasis_moves)
        
        return default_move
    
    def _generate_question_text(
        self,
        current_question: str,
        template: Dict[str, Any],
        cognitive_move: CognitiveMove,
        level: int,
        context: Optional[str]
    ) -> str:
        """Generate question text based on parameters."""
        # Extract key terms from current question
        key_terms = self._extract_key_terms(current_question)
        
        # Generate based on cognitive move and template focus
        move_templates = {
            CognitiveMove.DEFINE: [
                f"What exactly does '{key_terms[0] if key_terms else 'success'}' mean in this context?",
                f"How should we define the key terms and concepts involved?",
                f"What are the essential elements that must be understood first?"
            ],
            CognitiveMove.SCOPE: [
                f"What are the boundaries and constraints for this {template['focus']} approach?",
                f"What scope limitations should we consider?",
                f"Which aspects are in-scope vs out-of-scope for this approach?"
            ],
            CognitiveMove.QUANTIFY: [
                f"What metrics would help measure progress on this {template['focus']} approach?",
                f"How can we quantify the key variables and outcomes?",
                f"What thresholds or benchmarks should guide our decisions?"
            ],
            CognitiveMove.COMPARE: [
                f"How does this {template['focus']} approach compare to alternatives?",
                f"What are the trade-offs between different options?",
                f"Which factors are most important when comparing approaches?"
            ],
            CognitiveMove.SIMULATE: [
                f"What would happen if we pursued this {template['focus']} approach?",
                f"What scenarios should we model and test?",
                f"What are the likely outcomes and side effects?"
            ],
            CognitiveMove.DECIDE: [
                f"Based on the analysis, which {template['focus']} option should we choose?",
                f"What decision criteria should guide our final choice?",
                f"Which approach best balances all the factors?"
            ],
            CognitiveMove.COMMIT: [
                f"What specific actions will implement this {template['focus']} decision?",
                f"How do we commit to and execute this approach?",
                f"What are the next concrete steps to take?"
            ]
        }
        
        templates = move_templates.get(cognitive_move, ["What should we consider next?"])
        return random.choice(templates)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from question text."""
        # Simple extraction - get capitalized words and important terms
        import re
        
        # Find capitalized words (likely important terms)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Find quoted terms
        quoted = re.findall(r'"([^"]*)"', text)
        
        # Find important question words
        important_words = []
        for word in text.split():
            word_clean = word.strip('.,?!').lower()
            if len(word_clean) > 4 and word_clean not in ['should', 'would', 'could', 'might', 'where', 'when', 'what', 'how']:
                important_words.append(word_clean)
        
        return capitalized + quoted + important_words[:3]
    
    def _generate_delta_nuance(self, cognitive_move: CognitiveMove, template: Dict[str, Any]) -> str:
        """Generate delta nuance description."""
        base_nuance = self.move_progression.get_move_delta_nuance_template(cognitive_move)
        focus = template["focus"]
        
        nuance_map = {
            CognitiveMove.DEFINE: f"Establishes {focus}-focused definition",
            CognitiveMove.SCOPE: f"Adds {focus} boundary constraints", 
            CognitiveMove.QUANTIFY: f"Introduces {focus} measurement framework",
            CognitiveMove.COMPARE: f"Compares {focus} alternatives",
            CognitiveMove.SIMULATE: f"Models {focus} scenarios",
            CognitiveMove.DECIDE: f"Decides on {focus} approach",
            CognitiveMove.COMMIT: f"Commits to {focus} implementation"
        }
        
        return nuance_map.get(cognitive_move, f"Adds {focus} perspective")
    
    async def _stitch_lane_questions(self, questions: List[Question]) -> List[Question]:
        """Build proper builds_on relationships within a lane."""
        if not questions:
            return questions
        
        # Sort by level
        questions.sort(key=lambda q: q.level)
        
        # First question has no dependencies
        if questions:
            questions[0].builds_on = []
        
        # Each subsequent question builds on the previous one(s)
        for i in range(1, len(questions)):
            current = questions[i]
            
            # Find appropriate parents (usually previous level)
            parents = [q for q in questions[:i] if q.level == current.level - 1]
            
            if not parents:
                # If no direct parents, connect to highest available level
                available_parents = [q for q in questions[:i]]
                if available_parents:
                    parents = [max(available_parents, key=lambda q: q.level)]
            
            current.builds_on = [p.id for p in parents]
        
        return questions
    
    def _add_triggers_to_questions(self, questions: List[Question]) -> None:
        """Add triggers to appropriate questions."""
        for question in questions:
            # Add triggers based on cognitive move and content
            if question.cognitive_move == CognitiveMove.QUANTIFY:
                question.triggers.append(Trigger(
                    type=TriggerType.METRIC,
                    detail="When key metrics are defined and measurable"
                ))
            elif question.cognitive_move == CognitiveMove.SIMULATE:
                question.triggers.append(Trigger(
                    type=TriggerType.EVENT,
                    detail="When scenario modeling is needed"
                ))
            elif question.cognitive_move == CognitiveMove.DECIDE:
                question.triggers.append(Trigger(
                    type=TriggerType.ANSWER_CHANGE,
                    detail="When previous analysis changes decision criteria"
                ))
    
    def _mark_natural_endings(self, questions: List[Question]) -> None:
        """Mark questions that represent natural endings."""
        for question in questions:
            # Terminal cognitive moves are natural endings
            if self.move_progression.is_terminal_move(question.cognitive_move):
                question.natural_end = True
            
            # Low info gain questions at higher levels
            elif question.level >= 4 and question.expected_info_gain < 0.3:
                question.natural_end = True
    
    def _add_cross_links(self, lanes: List[Lane]) -> None:
        """Add cross-links between related questions in different lanes."""
        # Find questions with similar cognitive moves and levels
        for i, lane_a in enumerate(lanes):
            for j, lane_b in enumerate(lanes[i+1:], i+1):
                # Look for junction opportunities
                for q_a in lane_a.lane:
                    for q_b in lane_b.lane:
                        if (q_a.cognitive_move == q_b.cognitive_move and 
                            abs(q_a.level - q_b.level) <= 1 and
                            q_a.expected_info_gain > 0.5 and q_b.expected_info_gain > 0.5):
                            
                            # Add cross-link
                            cross_link = CrossLink(
                                from_id=q_a.id,
                                to_id=q_b.id,
                                type="junction"
                            )
                            lane_a.cross_links.append(cross_link)
                            break
    
    def _calculate_progression_quality(self, questions: List[Question]) -> float:
        """Calculate how well questions follow logical progression."""
        if len(questions) < 2:
            return 1.0
        
        quality_score = 0.0
        valid_progressions = 0
        
        for i in range(1, len(questions)):
            current = questions[i]
            for parent_id in current.builds_on:
                parent = next((q for q in questions if q.id == parent_id), None)
                if parent:
                    if self.move_progression.is_valid_progression(parent.cognitive_move, current.cognitive_move):
                        quality_score += 1.0
                    valid_progressions += 1
        
        return quality_score / max(valid_progressions, 1)
    
    async def _generate_follow_up_question(
        self, 
        node: Question, 
        next_move: CognitiveMove,
        user_answer: Optional[str]
    ) -> Optional[Question]:
        """Generate a follow-up question from a node."""
        # Incorporate user answer if provided
        context_addition = f" Given that {user_answer}," if user_answer else ""
        
        follow_up_text = f"{context_addition} {self._generate_move_question(next_move, node.text)}"
        
        follow_up = Question(
            id=f"Q{self._next_id()}",
            text=follow_up_text,
            level=node.level + 1,
            cognitive_move=next_move,
            builds_on=[node.id],
            delta_nuance=f"Builds on {node.id} with {next_move.value} perspective",
            expected_info_gain=self.scorer.calculate_info_gain(Question(
                id="temp", text=follow_up_text, level=node.level + 1,
                cognitive_move=next_move, builds_on=[], delta_nuance="", 
                expected_info_gain=0.0, confidence=0.0
            )),
            confidence=0.0,
            tags=["follow_up", next_move.value]
        )
        
        follow_up.confidence = self.scorer.calculate_confidence(follow_up)
        
        return follow_up
    
    async def _generate_continuation_question(
        self, 
        node: Question,
        level: int
    ) -> Optional[Question]:
        """Generate a continuation question from a node."""
        # Get next logical move
        next_moves = self.move_progression.get_next_moves(node.cognitive_move)
        if not next_moves:
            return None
        
        next_move = self.move_progression.suggest_next_move(node.cognitive_move)
        
        continuation_text = self._generate_move_question(next_move, node.text)
        
        continuation = Question(
            id=f"Q{self._next_id()}",
            text=continuation_text,
            level=level,
            cognitive_move=next_move,
            builds_on=[node.id],
            delta_nuance=f"Continues from {node.id} with {next_move.value}",
            expected_info_gain=0.0,
            confidence=0.0,
            tags=["continuation", next_move.value]
        )
        
        continuation.expected_info_gain = self.scorer.calculate_info_gain(continuation)
        continuation.confidence = self.scorer.calculate_confidence(continuation)
        
        return continuation
    
    def _generate_move_question(self, move: CognitiveMove, base_text: str) -> str:
        """Generate a question for a specific cognitive move."""
        move_templates = {
            CognitiveMove.DEFINE: "What exactly do we mean by the key terms here?",
            CognitiveMove.SCOPE: "What are the boundaries and constraints we should consider?",
            CognitiveMove.QUANTIFY: "How can we measure and track progress on this?",
            CognitiveMove.COMPARE: "What alternatives should we compare this against?",
            CognitiveMove.SIMULATE: "What scenarios should we model and test?",
            CognitiveMove.DECIDE: "Based on the analysis, what should we choose?",
            CognitiveMove.COMMIT: "What are the specific next steps to implement this?"
        }
        
        return move_templates.get(move, "What should we consider next?")
    
    def _next_id(self) -> int:
        """Get next question ID number."""
        self._question_counter += 1
        return self._question_counter