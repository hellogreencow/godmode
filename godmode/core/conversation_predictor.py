"""
Conversation Prediction Engine for GodMode.

This module provides predictive conversation intelligence that can:
- Predict the next 30 questions in a conversation
- Explore alternative conversation branches
- Reveal the 10 questions that could have led to the current thought
- Provide AI-driven reasoning type selection with explanations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

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
class PredictionResult:
    """Complete prediction result."""
    current_question: str
    future_questions: List[str]  # Next 30 predicted questions
    alternative_branches: List[ConversationBranch]  # Alternative conversation paths
    origin_questions: List[str]  # 10 questions that could have led here
    selected_reasoning_type: str
    reasoning_explanation: str
    confidence_score: float
    prediction_metadata: Dict[str, Any]


class ConversationPredictor:
    """AI-powered conversation prediction engine."""

    def __init__(self, openrouter_config: Optional[OpenRouterConfig] = None):
        """Initialize the conversation predictor."""
        self.openrouter = OpenRouterIntegration(openrouter_config)
        self.conversation_history: List[ConversationNode] = []
        self.prediction_cache: Dict[str, PredictionResult] = {}

    async def predict_conversation(
        self,
        current_question: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        max_future_questions: int = 30,
        max_branches: int = 5,
        max_origin_questions: int = 10
    ) -> PredictionResult:
        """
        Predict the complete conversation trajectory.

        Args:
            current_question: The current question being asked
            conversation_context: Previous conversation history
            max_future_questions: Number of future questions to predict
            max_branches: Number of alternative branches to explore
            max_origin_questions: Number of origin questions to reveal

        Returns:
            Complete prediction result with all analysis
        """

        # Check cache first
        cache_key = f"{current_question}_{len(conversation_context or [])}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        logger.info(f"🔮 Predicting conversation for: {current_question[:100]}...")

        # Step 1: Analyze and select reasoning type
        reasoning_analysis = await self._analyze_reasoning_type(current_question)

        # Step 2: Predict future questions
        future_questions = await self._predict_future_questions(
            current_question,
            conversation_context,
            max_future_questions,
            reasoning_analysis['type']
        )

        # Step 3: Explore alternative branches
        alternative_branches = await self._explore_branches(
            current_question,
            conversation_context,
            max_branches
        )

        # Step 4: Reveal origin questions
        origin_questions = await self._reveal_origin_questions(
            current_question,
            conversation_context,
            max_origin_questions
        )

        # Step 5: Calculate overall confidence
        confidence_score = self._calculate_prediction_confidence(
            future_questions, alternative_branches, origin_questions
        )

        result = PredictionResult(
            current_question=current_question,
            future_questions=future_questions,
            alternative_branches=alternative_branches,
            origin_questions=origin_questions,
            selected_reasoning_type=reasoning_analysis['type'],
            reasoning_explanation=reasoning_analysis['explanation'],
            confidence_score=confidence_score,
            prediction_metadata={
                'timestamp': datetime.now().isoformat(),
                'model_used': 'anthropic/claude-3.5-sonnet',
                'prediction_depth': max_future_questions,
                'branches_explored': max_branches,
                'processing_time': 0.0  # Will be set by caller
            }
        )

        # Cache the result
        self.prediction_cache[cache_key] = result

        logger.info(f"✅ Prediction complete - {len(future_questions)} future questions, {len(alternative_branches)} branches, {len(origin_questions)} origins")

        return result

    async def _analyze_reasoning_type(self, question: str) -> Dict[str, str]:
        """AI-driven reasoning type selection with explanation."""

        system_prompt = """You are an expert conversation analyst. Analyze the given question and determine the most appropriate reasoning type from: hierarchical, forward, backward, analogical, causal, temporal, abductive, creative.

For each reasoning type, provide:
1. Why this type fits the question
2. What insights it would reveal
3. Expected outcomes

Then select the BEST type and explain your choice comprehensively."""

        user_prompt = f"""Analyze this question and select the optimal reasoning type:

QUESTION: {question}

Provide your analysis and final recommendation."""

        result = await self.openrouter.generate_reasoning(
            problem=Problem(
                title="Reasoning Type Analysis",
                description=user_prompt,
                domain="conversation_analysis",
                problem_type="classification"
            ),
            system_prompt=system_prompt,
            max_tokens=1000,
            temperature=0.3
        )

        if result['success']:
            # Parse the response to extract reasoning type and explanation
            content = result['content']
            return self._parse_reasoning_analysis(content)
        else:
            # Fallback to hierarchical reasoning
            return {
                'type': 'hierarchical',
                'explanation': 'Using hierarchical reasoning as the default approach for structured problem analysis.'
            }

    def _parse_reasoning_analysis(self, content: str) -> Dict[str, str]:
        """Parse AI response to extract reasoning type and explanation."""

        # Look for explicit type selection in the response
        type_patterns = [
            r'(?:selected|chosen|recommended|best).*?(hierarchical|forward|backward|analogical|causal|temporal|abductive|creative)',
            r'(?:reasoning type|approach).*?[:\-]\s*(hierarchical|forward|backward|analogical|causal|temporal|abductive|creative)',
            r'(hierarchical|forward|backward|analogical|causal|temporal|abductive|creative).*?(?:reasoning|approach)'
        ]

        selected_type = 'hierarchical'  # default

        for pattern in type_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                selected_type = match.group(1).lower()
                break

        # Extract explanation (everything after the type selection)
        explanation = content
        if ':' in content:
            parts = content.split(':', 1)
            if len(parts) > 1:
                explanation = parts[1].strip()

        return {
            'type': selected_type,
            'explanation': explanation
        }

    async def _predict_future_questions(
        self,
        current_question: str,
        context: Optional[List[Dict[str, Any]]],
        max_questions: int,
        reasoning_type: str
    ) -> List[str]:
        """Predict the next N questions in the conversation."""

        context_str = ""
        if context:
            context_str = "\n".join([f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}" for item in context[-5:]])

        system_prompt = f"""You are a conversation prediction expert. Predict the next {max_questions} questions that would naturally follow from the current question, considering the conversation context and using {reasoning_type} reasoning.

Think step-by-step about:
1. What information gaps exist
2. What clarifications would be needed
3. What follow-up questions would deepen understanding
4. What related topics would naturally emerge
5. What practical applications or implementations would be relevant

Provide exactly {max_questions} questions, each on a new line, starting with a number."""

        user_prompt = f"""Current Question: {current_question}

Recent Context:
{context_str}

Predict the next {max_questions} questions that would naturally follow."""

        result = await self.openrouter.generate_reasoning(
            problem=Problem(
                title="Future Question Prediction",
                description=user_prompt,
                domain="conversation_prediction",
                problem_type="prediction"
            ),
            system_prompt=system_prompt,
            max_tokens=1500,
            temperature=0.7
        )

        if result['success']:
            return self._parse_numbered_list(result['content'], max_questions)
        else:
            return [f"Unable to predict future questions: {result.get('error', 'Unknown error')}"]

    async def _explore_branches(
        self,
        current_question: str,
        context: Optional[List[Dict[str, Any]]],
        max_branches: int
    ) -> List[ConversationBranch]:
        """Explore alternative conversation branches."""

        context_str = ""
        if context:
            context_str = "\n".join([f"Q: {item.get('question', '')}" for item in context[-3:]])

        system_prompt = f"""You are a conversation strategist. Explore {max_branches} alternative conversation branches that could emerge from the current question.

For each branch, consider:
1. Different interpretations of the question
2. Alternative approaches or perspectives
3. Related but distinct topics
4. Practical vs theoretical angles
5. Short-term vs long-term implications

Format each branch as:
BRANCH X: [Brief title]
Path: [2-3 key questions that define this branch]
Outcome: [What this branch leads to]
Probability: [High/Medium/Low]"""

        user_prompt = f"""Current Question: {current_question}

Context: {context_str}

Explore {max_branches} alternative conversation branches."""

        result = await self.openrouter.generate_reasoning(
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

        if result['success']:
            return self._parse_branches(result['content'], max_branches)
        else:
            return []

    async def _reveal_origin_questions(
        self,
        current_question: str,
        context: Optional[List[Dict[str, Any]]],
        max_origins: int
    ) -> List[str]:
        """Reveal questions that could have led to the current thought."""

        system_prompt = f"""You are a conversation archaeologist. Trace backwards to find the {max_origins} questions that could have naturally led to the current question.

Think about:
1. Foundational concepts that must be understood first
2. Prerequisites and background knowledge
3. Progressive levels of complexity
4. Logical dependencies and sequences
5. Knowledge gaps that would need filling

Provide exactly {max_origins} questions that build up to the current question."""

        user_prompt = f"""Current Question: {current_question}

What {max_origins} questions would someone need to ask (in sequence) to naturally arrive at this current question?"""

        result = await self.openrouter.generate_reasoning(
            problem=Problem(
                title="Origin Question Analysis",
                description=user_prompt,
                domain="conversation_history",
                problem_type="retrospective"
            ),
            system_prompt=system_prompt,
            max_tokens=1000,
            temperature=0.6
        )

        if result['success']:
            return self._parse_numbered_list(result['content'], max_origins)
        else:
            return [f"Unable to trace origin questions: {result.get('error', 'Unknown error')}"]

    def _parse_numbered_list(self, content: str, max_items: int) -> List[str]:
        """Parse numbered list from AI response."""
        lines = content.strip().split('\n')
        questions = []

        for line in lines:
            line = line.strip()
            # Look for numbered items: 1. Question, 1) Question, etc.
            if re.match(r'^\d+[\.\)\-\s]', line):
                # Remove the numbering
                question = re.sub(r'^\d+[\.\)\-\s]+', '', line).strip()
                if question and len(question) > 10:  # Filter out too short items
                    questions.append(question)
                    if len(questions) >= max_items:
                        break

        # If we didn't get enough numbered items, try to extract any questions
        if len(questions) < max_items:
            # Look for question marks
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                if '?' in sentence and len(sentence.strip()) > 20:
                    questions.append(sentence.strip())
                    if len(questions) >= max_items:
                        break

        return questions[:max_items] if questions else ["Unable to extract questions from response"]

    def _parse_branches(self, content: str, max_branches: int) -> List[ConversationBranch]:
        """Parse conversation branches from AI response."""
        branches = []
        sections = re.split(r'BRANCH \d+:', content, flags=re.IGNORECASE)

        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            if i > max_branches:
                break

            try:
                # Extract branch information
                lines = section.strip().split('\n')
                title = lines[0].strip() if lines else f"Branch {i}"

                path = ""
                outcome = ""
                probability = "Medium"

                for line in lines:
                    if line.lower().startswith('path:'):
                        path = line.split(':', 1)[1].strip()
                    elif line.lower().startswith('outcome:'):
                        outcome = line.split(':', 1)[1].strip()
                    elif 'probability:' in line.lower():
                        prob_match = re.search(r'probability:\s*(high|medium|low)', line, re.IGNORECASE)
                        if prob_match:
                            probability = prob_match.group(1).title()

                # Create branch
                branch = ConversationBranch(
                    branch_id=f"branch_{i}",
                    nodes=[],  # Would be populated with actual nodes
                    probability=0.6 if probability == "High" else 0.4 if probability == "Medium" else 0.2,
                    reasoning_path=path,
                    outcome_prediction=outcome,
                    key_decision_points=[title]  # Simplified
                )

                branches.append(branch)

            except Exception as e:
                logger.warning(f"Error parsing branch {i}: {e}")
                continue

        return branches

    def _calculate_prediction_confidence(
        self,
        future_questions: List[str],
        branches: List[ConversationBranch],
        origin_questions: List[str]
    ) -> float:
        """Calculate overall confidence in predictions."""

        base_confidence = 0.7  # Base confidence

        # Factors that increase confidence
        if len(future_questions) >= 10:
            base_confidence += 0.1
        if len(branches) >= 3:
            base_confidence += 0.1
        if len(origin_questions) >= 5:
            base_confidence += 0.1

        # Factors that decrease confidence
        if any("unable to" in q.lower() or "error" in q.lower() for q in future_questions):
            base_confidence -= 0.2
        if len(branches) == 0:
            base_confidence -= 0.2

        return max(0.1, min(1.0, base_confidence))

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models for conversation prediction."""
        try:
            models = await self.openrouter.get_available_models()
            return [
                {
                    'id': model.id,
                    'name': model.name,
                    'provider': model.provider,
                    'capabilities': model.capabilities
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []

    async def close(self):
        """Clean up resources."""
        await self.openrouter.close()


# Global predictor instance
conversation_predictor = ConversationPredictor()
