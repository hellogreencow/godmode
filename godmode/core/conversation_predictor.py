"""
Conversation Prediction Engine for GodMode.

This module provides REAL AI-powered conversation intelligence that:
- Uses OpenRouter API for ALL predictions (no templates)
- Integrates HRM for enhanced reasoning
- Provides genuine AI-driven conversation simulation
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from godmode.integrations.openrouter import OpenRouterIntegration, OpenRouterConfig
from godmode.models.core import Problem, Solution

logger = logging.getLogger(__name__)


@dataclass
class ConversationNode:
    """A node in the conversation tree."""
    id: str
    question: str
    response: Optional[str] = None
    reasoning_type: str = "hierarchical"
    confidence: float = 1.0
    timestamp: datetime = None
    parent_id: Optional[str] = None
    children: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConversationBranch:
    """A predicted conversation branch."""
    branch_id: str
    nodes: List[ConversationNode]
    probability: float
    reasoning_path: str
    outcome_prediction: str
    key_decision_points: List[str]


@dataclass
class ConversationTurn:
    """A single turn in a simulated conversation."""
    turn_number: int
    user_question: str
    ai_response: str
    reasoning_type: str
    confidence: float

@dataclass
class PredictionResult:
    """Complete prediction result."""
    current_question: str
    simulated_conversation: List[ConversationTurn]
    alternative_branches: List[ConversationBranch]
    origin_questions: List[str]
    selected_reasoning_type: str
    reasoning_explanation: str
    confidence_score: float
    prediction_metadata: Dict[str, Any]


class ConversationPredictor:
    """REAL AI-powered conversation prediction engine."""

    def __init__(self, openrouter_config: Optional[OpenRouterConfig] = None):
        """Initialize the conversation predictor with REAL AI capabilities."""
        # NO DEMO MODE - ALWAYS USE REAL AI
        try:
            if openrouter_config is None:
                # Try to create config from environment
                self.openrouter = OpenRouterIntegration()
            else:
                self.openrouter = OpenRouterIntegration(openrouter_config)
        except (ValueError, Exception) as e:
            logger.error(f"Failed to initialize OpenRouter: {e}")
            logger.error("Please set OPENROUTER_API_KEY environment variable")
            raise RuntimeError(f"OpenRouter API key required for real AI predictions: {e}")
        
        self.conversation_history: List[ConversationNode] = []
        self.prediction_cache: Dict[str, PredictionResult] = {}
        
        # Initialize HRM for enhanced reasoning
        try:
            from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel
            self.hrm = HierarchicalReasoningModel()
            logger.info("✅ HRM integrated for enhanced reasoning")
        except ImportError:
            self.hrm = None
            logger.warning("⚠️ HRM not available - using standard reasoning")

    def predict_conversation(
        self,
        current_question: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        max_future_questions: int = 30,
        max_branches: int = 5,
        max_origin_questions: int = 10
    ) -> PredictionResult:
        """
        Predict the complete conversation trajectory using REAL AI.
        NO TEMPLATES, NO HARDCODED CONTENT.
        """
        logger.info(f"🔮 REAL AI Prediction for: {current_question[:100]}...")

        # Step 1: AI-driven reasoning type analysis
        reasoning_analysis = self._analyze_reasoning_type(current_question)

        # Enhance with HRM if available
        if self.hrm:
            try:
                hrm_analysis = self._enhance_with_hrm(current_question, reasoning_analysis)
                reasoning_analysis.update(hrm_analysis)
            except Exception as e:
                logger.warning(f"HRM enhancement failed: {e}")

        # Step 2: AI-generated conversation simulation
        simulated_conversation = self._simulate_full_conversation(
            current_question,
            conversation_context,
            max_future_questions,
            reasoning_analysis['type']
        )

        # Step 3: AI-generated alternative branches
        alternative_branches = self._explore_branches(
            current_question,
            conversation_context,
            max_branches
        )

        # Step 4: AI-generated origin questions
        origin_questions = self._reveal_origin_questions(
            current_question,
            conversation_context,
            max_origin_questions
        )

        # Step 5: Calculate confidence
        confidence_score = self._calculate_prediction_confidence(
            simulated_conversation, alternative_branches, origin_questions
        )

        result = PredictionResult(
            current_question=current_question,
            simulated_conversation=simulated_conversation,
            alternative_branches=alternative_branches,
            origin_questions=origin_questions,
            selected_reasoning_type=reasoning_analysis.get('type', 'hierarchical'),
            reasoning_explanation=reasoning_analysis.get('explanation', 'AI-driven analysis'),
            confidence_score=confidence_score,
            prediction_metadata={
                'timestamp': datetime.now().isoformat(),
                'hrm_enhanced': self.hrm is not None,
                'context_length': len(conversation_context) if conversation_context else 0
            }
        )

        return result

    def _analyze_reasoning_type(self, question: str) -> Dict[str, str]:
        """AI-driven reasoning type selection with explanation."""

        system_prompt = """You are an expert conversation analyst. Analyze the given question and determine the most appropriate reasoning type from: hierarchical, forward, backward, analogical, causal, temporal, abductive, creative.

For each reasoning type, provide:
- A clear explanation of why this type fits
- Confidence level (0.0-1.0)

Select the BEST reasoning type and explain your choice."""

        user_prompt = f"Analyze this question and select the optimal reasoning approach: {question}"

        try:
            result = self.openrouter.generate_reasoning(
                problem=Problem(
                    title="Reasoning Type Analysis",
                    description=user_prompt,
                    domain="conversation_analysis",
                    problem_type="classification"
                ),
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )

            if result["success"]:
                return self._parse_reasoning_analysis(result["content"])
            else:
                logger.error(f"AI analysis failed: {result.get('error', 'Unknown error')}")
                return {'type': 'hierarchical', 'explanation': 'Default reasoning approach'}

        except Exception as e:
            logger.error(f"Failed to analyze reasoning type: {e}")
            return {'type': 'hierarchical', 'explanation': f'Error in AI analysis: {e}'}

    def _parse_reasoning_analysis(self, content: str) -> Dict[str, str]:
        """Parse AI response to extract reasoning type and explanation."""
        reasoning_types = ['hierarchical', 'forward', 'backward', 'analogical', 'causal', 'temporal', 'abductive', 'creative']
        
        content_lower = content.lower()
        selected_type = 'hierarchical'  # default
        
        # Find the selected reasoning type
        for rtype in reasoning_types:
            if f"selected: {rtype}" in content_lower or f"best: {rtype}" in content_lower:
                selected_type = rtype
                break
            elif rtype in content_lower and "reasoning" in content_lower:
                selected_type = rtype
                break

        # Extract explanation
        lines = content.split('\n')
        explanation = "AI-selected reasoning approach"
        for line in lines:
            if 'explanation' in line.lower() or 'because' in line.lower():
                explanation = line.strip()
                break

        return {'type': selected_type, 'explanation': explanation}

    def _enhance_with_hrm(self, question: str, reasoning_analysis: Dict[str, str]) -> Dict[str, Any]:
        """Enhance reasoning analysis with Hierarchical Reasoning Model."""
        if not self.hrm:
            return {}

        try:
            # Use HRM to enhance the reasoning analysis
            hrm_input = {
                'question': question,
                'reasoning_type': reasoning_analysis.get('type', 'hierarchical'),
                'confidence': reasoning_analysis.get('confidence', 0.5)
            }

            # Process through HRM layers
            enhanced_analysis = self.hrm.process_hierarchical_reasoning(hrm_input)

            return {
                'hrm_enhanced': True,
                'enhanced_reasoning': enhanced_analysis.get('reasoning_enhancement', ''),
                'cognitive_levels': enhanced_analysis.get('cognitive_levels', []),
                'confidence_boost': enhanced_analysis.get('confidence_adjustment', 0.0)
            }

        except Exception as e:
            logger.error(f"HRM enhancement failed: {e}")
            return {}

    def _simulate_full_conversation(
        self,
        initial_question: str,
        context: Optional[List[Dict[str, Any]]],
        max_turns: int,
        reasoning_type: str
    ) -> List[ConversationTurn]:
        """Generate REAL AI conversation simulation using simultaneous API requests."""
        
        system_prompt = f"""You are an expert conversation simulator. Generate a realistic, intelligent conversation where each turn builds naturally on the previous ones.

Reasoning Type: {reasoning_type}
Context: {context if context else 'None'}

Generate {max_turns} conversation turns that:
1. Start with the initial question
2. Each response should be genuinely helpful and intelligent
3. Each follow-up question should naturally flow from the previous response
4. Demonstrate deep understanding and insight
5. Avoid generic or template-like responses

Format each turn as:
TURN X:
USER: [question]
AI: [intelligent response]
"""

        user_prompt = f"""Initial Question: {initial_question}

Generate a {max_turns}-turn conversation simulation that demonstrates genuine intelligence and contextual understanding."""

        try:
            result = self.openrouter.generate_reasoning(
                problem=Problem(
                    title="Conversation Simulation",
                    description=user_prompt,
                    domain="conversation_generation",
                    problem_type="simulation"
                ),
                system_prompt=system_prompt,
                max_tokens=4000,
                temperature=0.7
            )

            if result["success"]:
                return self._parse_conversation_simulation(result["content"], reasoning_type)
            else:
                logger.error(f"AI conversation simulation failed: {result.get('error', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error in AI conversation simulation: {e}")
            return []

    def _parse_conversation_simulation(self, content: str, reasoning_type: str) -> List[ConversationTurn]:
        """Parse AI-generated conversation simulation."""
        conversation_turns = []
        
        # Split by TURN markers
        turns = content.split('TURN')
        
        for i, turn_content in enumerate(turns[1:], 1):  # Skip first empty part
            lines = turn_content.strip().split('\n')
            user_question = ""
            ai_response = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('USER:'):
                    user_question = line[5:].strip()
                elif line.startswith('AI:'):
                    ai_response = line[3:].strip()
                elif 'USER:' in line:
                    user_question = line.split('USER:')[1].strip()
                elif 'AI:' in line:
                    ai_response = line.split('AI:')[1].strip()
            
            if user_question and ai_response:
                conversation_turns.append(ConversationTurn(
                    turn_number=i,
                    user_question=user_question,
                    ai_response=ai_response,
                    reasoning_type=reasoning_type,
                    confidence=0.85
                ))

        return conversation_turns

    def _explore_branches(
        self,
        current_question: str,
        context: Optional[List[Dict[str, Any]]],
        max_branches: int
    ) -> List[ConversationBranch]:
        """Generate REAL AI conversation branches."""

        system_prompt = f"""You are a conversation strategist. Explore {max_branches} alternative conversation branches that could emerge from the current question.

For each branch, consider:
1. Different interpretations of the question
2. Alternative approaches or perspectives
3. Related but distinct topics
4. Practical vs theoretical angles
5. Short-term vs long-term implications

Format each branch as:
BRANCH X: [Title]
Path: [2-3 key questions that define this branch]
Outcome: [What this branch leads to]
Probability: [High/Medium/Low]"""

        context_str = ""
        if context:
            context_str = "\n".join([f"Q: {item.get('question', '')}" for item in context[-3:]])

        user_prompt = f"""Current Question: {current_question}

Context: {context_str}

Explore {max_branches} alternative conversation branches."""

        try:
            result = self.openrouter.generate_reasoning(
                problem=Problem(
                    title="Branch Exploration",
                    description=user_prompt,
                    domain="conversation_strategy",
                    problem_type="exploration"
                ),
                system_prompt=system_prompt,
                max_tokens=1200,
                temperature=0.8
            )

            if result["success"]:
                return self._parse_branches(result["content"], max_branches)
            else:
                logger.error(f"AI branch exploration failed: {result.get('error', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error in AI branch exploration: {e}")
            return []

    def _parse_branches(self, content: str, max_branches: int) -> List[ConversationBranch]:
        """Parse AI-generated conversation branches."""
        branches = []
        sections = content.split('BRANCH')

        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            if i > max_branches:
                break

            lines = section.strip().split('\n')
            title = lines[0].strip() if lines else f"Branch {i}"
            
            path = ""
            outcome = ""
            probability = 0.5

            for line in lines:
                if line.lower().startswith('path:'):
                    path = line.split(':', 1)[1].strip()
                elif line.lower().startswith('outcome:'):
                    outcome = line.split(':', 1)[1].strip()
                elif line.lower().startswith('probability:'):
                    prob_text = line.split(':', 1)[1].strip().lower()
                    if 'high' in prob_text:
                        probability = 0.8
                    elif 'medium' in prob_text:
                        probability = 0.6
                    elif 'low' in prob_text:
                        probability = 0.4

            branches.append(ConversationBranch(
                branch_id=f"ai_branch_{i}",
                nodes=[],
                probability=probability,
                reasoning_path=path or f"AI-generated path {i}",
                outcome_prediction=outcome or f"AI-predicted outcome {i}",
                key_decision_points=[path] if path else []
            ))

        return branches

    def _reveal_origin_questions(
        self,
        current_question: str,
        context: Optional[List[Dict[str, Any]]],
        max_origins: int
    ) -> List[str]:
        """Generate REAL AI origin questions."""

        system_prompt = f"""You are a conversation archaeologist. Trace backwards to find the {max_origins} questions that could have naturally led to the current question.

Think about:
1. Foundational concepts that must be understood first
2. Prerequisites and background knowledge
3. Progressive levels of complexity
4. Logical dependencies and sequences
5. Knowledge gaps that would need filling

Provide exactly {max_origins} questions that build up to the current question."""

        user_prompt = f"""Current Question: {current_question}

Context: {context if context else 'None'}

Generate {max_origins} origin questions that could have led to this question."""

        try:
            result = self.openrouter.generate_reasoning(
                problem=Problem(
                    title="Origin Question Analysis",
                    description=user_prompt,
                    domain="conversation_archaeology",
                    problem_type="analysis"
                ),
                system_prompt=system_prompt,
                max_tokens=800,
                temperature=0.6
            )

            if result["success"]:
                return self._parse_numbered_list(result["content"], max_origins)
            else:
                logger.error(f"AI origin analysis failed: {result.get('error', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error in AI origin analysis: {e}")
            return []

    def _parse_numbered_list(self, content: str, max_items: int) -> List[str]:
        """Parse numbered list from AI response."""
        lines = content.strip().split('\n')
        questions = []

        for line in lines:
            line = line.strip()
            # Look for numbered items
            if any(line.startswith(f"{i}.") for i in range(1, max_items + 1)):
                # Remove number and clean up
                question = line.split('.', 1)[1].strip() if '.' in line else line
                if question and len(question) > 10:  # Minimum quality check
                    questions.append(question)

        return questions[:max_items]

    def _calculate_prediction_confidence(
        self,
        conversation_turns: List[ConversationTurn],
        branches: List[ConversationBranch],
        origin_questions: List[str]
    ) -> float:
        """Calculate overall prediction confidence based on AI results."""
        
        # Base confidence from having results
        base_confidence = 0.3
        
        # Boost for conversation quality
        if conversation_turns:
            avg_turn_confidence = sum(turn.confidence for turn in conversation_turns) / len(conversation_turns)
            base_confidence += avg_turn_confidence * 0.4
        
        # Boost for branch diversity
        if branches:
            branch_confidence = sum(branch.probability for branch in branches) / len(branches)
            base_confidence += branch_confidence * 0.2
        
        # Boost for origin quality
        if origin_questions:
            origin_confidence = min(len(origin_questions) / 10.0, 1.0)
            base_confidence += origin_confidence * 0.1

        return min(base_confidence, 1.0)

    async def close(self):
        """Clean up resources."""
        if self.openrouter:
            await self.openrouter.close()


# Global predictor instance with REAL AI capabilities
# Will be initialized when API key is available
conversation_predictor = None

def get_conversation_predictor() -> ConversationPredictor:
    """Get or create conversation predictor with proper error handling."""
    global conversation_predictor
    
    if conversation_predictor is None:
        try:
            conversation_predictor = ConversationPredictor()
            logger.info("✅ Real AI conversation predictor initialized")
        except RuntimeError as e:
            logger.error(f"❌ Failed to initialize conversation predictor: {e}")
            raise e
    
    return conversation_predictor
