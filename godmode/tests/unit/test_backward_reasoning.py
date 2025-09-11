"""Test backward reasoning (PRIOR ladders generation)."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from godmode.core.reasoning.backward import BackwardReasoning
from godmode.models.core import Question, CognitiveMove
from godmode.models.commands import Budgets


class TestBackwardReasoning:
    """Test BackwardReasoning class."""
    
    @pytest.fixture
    def backward_reasoning(self):
        """Create BackwardReasoning instance."""
        return BackwardReasoning()
    
    @pytest.fixture
    def sample_budgets(self):
        """Create sample budgets."""
        return Budgets(
            beam_width=4,
            depth_back=3,
            depth_fwd=5,
            prune_if_info_gain_below=0.2
        )
    
    @pytest.mark.asyncio
    async def test_enumerate_priors_basic(self, backward_reasoning, sample_budgets):
        """Test basic prior enumeration."""
        current_question = "How should we improve customer satisfaction?"
        context = "We're a SaaS company with 1000+ users"
        
        priors = await backward_reasoning.enumerate_priors(
            current_question, context, sample_budgets
        )
        
        # Should generate some prior questions
        assert len(priors) > 0
        assert all(isinstance(q, Question) for q in priors)
        
        # Should have level 1 questions (premises)
        level_1_questions = [q for q in priors if q.level == 1]
        assert len(level_1_questions) > 0
        
        # Level 1 questions should have empty builds_on
        for q in level_1_questions:
            assert q.builds_on == []
    
    @pytest.mark.asyncio
    async def test_extract_premises(self, backward_reasoning):
        """Test premise extraction."""
        question = "What is the best way to increase revenue?"
        
        premises = await backward_reasoning._extract_premises(question, None)
        
        assert len(premises) > 0
        assert any("success" in premise.lower() for premise in premises)
        assert any("constraint" in premise.lower() for premise in premises)
    
    @pytest.mark.asyncio
    async def test_rerank_candidates(self, backward_reasoning):
        """Test candidate reranking."""
        candidates = [
            Question(
                id="QP1",
                text="What does success mean?",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Defines success",
                expected_info_gain=0.0,
                confidence=0.0
            ),
            Question(
                id="QP2",
                text="What are the constraints?",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=["QP1"],
                delta_nuance="Adds constraints",
                expected_info_gain=0.0,
                confidence=0.0
            )
        ]
        
        ranked = await backward_reasoning.rerank_candidates(candidates)
        
        # Should return same number of candidates
        assert len(ranked) == len(candidates)
        
        # Should have calculated scores
        for candidate in ranked:
            assert 0.0 <= candidate.expected_info_gain <= 1.0
            assert 0.0 <= candidate.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_stitch_ladder(self, backward_reasoning):
        """Test ladder stitching."""
        candidates = [
            Question(
                id="QP1",
                text="What does success mean?",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Defines success",
                expected_info_gain=0.8,
                confidence=0.7
            ),
            Question(
                id="QP2",
                text="What are the constraints?",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=[],  # Will be filled by stitching
                delta_nuance="Adds constraints",
                expected_info_gain=0.6,
                confidence=0.5
            )
        ]
        
        stitched = await backward_reasoning.stitch_ladder(candidates)
        
        assert len(stitched) == 2
        
        # Level 1 should have no dependencies
        level_1 = next(q for q in stitched if q.level == 1)
        assert level_1.builds_on == []
        
        # Level 2 should have dependencies
        level_2 = next(q for q in stitched if q.level == 2)
        assert len(level_2.builds_on) > 0
    
    def test_generate_premise_question(self, backward_reasoning):
        """Test premise question generation."""
        original = "How can we optimize performance?"
        
        premise = backward_reasoning._generate_premise_question(
            "definition_of_optimality", original
        )
        
        assert isinstance(premise, str)
        assert len(premise) > 0
        assert "optimal" in premise.lower() or "best" in premise.lower()
    
    @pytest.mark.asyncio
    async def test_generate_seed_ladder(self, backward_reasoning, sample_budgets):
        """Test seed ladder generation."""
        premise = "What does success mean in this context?"
        
        ladder = await backward_reasoning._generate_seed_ladder(premise, sample_budgets)
        
        # Should generate multiple levels based on depth_back
        assert len(ladder) <= sample_budgets.depth_back
        assert len(ladder) >= 1
        
        # Should start with level 1
        assert ladder[0].level == 1
        assert ladder[0].cognitive_move == CognitiveMove.DEFINE
        
        # Should have proper progression
        for i in range(1, len(ladder)):
            assert ladder[i].level > ladder[i-1].level
            assert ladder[i-1].id in ladder[i].builds_on
    
    def test_has_logical_dependency(self, backward_reasoning):
        """Test logical dependency detection."""
        parent = Question(
            id="QP1",
            text="What does customer success mean?",
            level=1,
            cognitive_move=CognitiveMove.DEFINE,
            builds_on=[],
            delta_nuance="Defines customer success",
            expected_info_gain=0.8,
            confidence=0.7
        )
        
        child = Question(
            id="QP2",
            text="How do we measure customer success?",
            level=2,
            cognitive_move=CognitiveMove.QUANTIFY,
            builds_on=[],
            delta_nuance="Measures customer success",
            expected_info_gain=0.6,
            confidence=0.5
        )
        
        # Should detect dependency due to shared concepts and valid progression
        has_dependency = backward_reasoning._has_logical_dependency(child, parent)
        assert has_dependency is True
        
        # Test with unrelated questions
        unrelated_parent = Question(
            id="QP3",
            text="What is the weather like?",
            level=1,
            cognitive_move=CognitiveMove.DEFINE,
            builds_on=[],
            delta_nuance="Defines weather",
            expected_info_gain=0.5,
            confidence=0.5
        )
        
        has_dependency = backward_reasoning._has_logical_dependency(child, unrelated_parent)
        assert has_dependency is False
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, backward_reasoning, sample_budgets):
        """Test handling of empty inputs."""
        # Empty question
        priors = await backward_reasoning.enumerate_priors("", None, sample_budgets)
        assert isinstance(priors, list)
        
        # Empty candidates for reranking
        ranked = await backward_reasoning.rerank_candidates([])
        assert ranked == []
        
        # Empty candidates for stitching
        stitched = await backward_reasoning.stitch_ladder([])
        assert stitched == []
    
    @pytest.mark.asyncio
    async def test_premise_patterns(self, backward_reasoning):
        """Test different premise extraction patterns."""
        test_cases = [
            ("What is the best approach?", ["definition_of_optimality"]),
            ("When should we implement this?", ["scope_constraints"]),
            ("How much will this cost?", ["comparison_metrics"]),
            ("If we proceed, what happens?", ["conditional_assumptions"])
        ]
        
        for question, expected_patterns in test_cases:
            premises = await backward_reasoning._extract_premises(question, None)
            
            # Should extract some premises
            assert len(premises) > 0
            
            # Should include fundamental premises
            assert any("success" in premise.lower() for premise in premises)
    
    def test_question_counter_increment(self, backward_reasoning):
        """Test that question counter increments properly."""
        initial_counter = backward_reasoning._question_counter
        
        id1 = backward_reasoning._next_id()
        assert id1 == initial_counter + 1
        
        id2 = backward_reasoning._next_id()
        assert id2 == initial_counter + 2
    
    @pytest.mark.asyncio
    async def test_budgets_respected(self, backward_reasoning):
        """Test that budgets are respected during generation."""
        limited_budgets = Budgets(
            beam_width=2,  # Limited beam width
            depth_back=2,  # Limited depth
            depth_fwd=3
        )
        
        question = "How can we improve our complex multi-faceted business operations?"
        
        priors = await backward_reasoning.enumerate_priors(question, None, limited_budgets)
        
        # Should respect beam width (number of premise seeds)
        # Each seed generates multiple levels, so total may be > beam_width
        assert len(priors) >= limited_budgets.beam_width
        
        # Should respect depth limit
        max_level = max(q.level for q in priors) if priors else 0
        assert max_level <= limited_budgets.depth_back