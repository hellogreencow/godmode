"""Test forward reasoning (FUTURE ladders generation)."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from godmode.core.reasoning.forward import ForwardReasoning
from godmode.models.core import Question, Lane, CognitiveMove, TriggerType
from godmode.models.commands import Budgets


class TestForwardReasoning:
    """Test ForwardReasoning class."""
    
    @pytest.fixture
    def forward_reasoning(self):
        """Create ForwardReasoning instance."""
        return ForwardReasoning()
    
    @pytest.fixture
    def sample_budgets(self):
        """Create sample budgets."""
        return Budgets(
            beam_width=4,
            depth_back=3,
            depth_fwd=4,
            prune_if_info_gain_below=0.2
        )
    
    @pytest.mark.asyncio
    async def test_enumerate_futures_basic(self, forward_reasoning, sample_budgets):
        """Test basic future enumeration."""
        current_question = "How should we scale our engineering team?"
        context = "We're a growing startup with 50 employees"
        
        futures = await forward_reasoning.enumerate_futures(
            current_question, context, sample_budgets
        )
        
        # Should generate 3-5 scenario lanes
        assert 3 <= len(futures) <= 5
        
        # Each future should have required fields
        for future in futures:
            assert "id" in future
            assert "name" in future
            assert "description" in future
            assert "questions" in future
            assert "template" in future
            
            # Should have questions
            assert len(future["questions"]) > 0
            
            # All questions should be Question objects
            assert all(isinstance(q, Question) for q in future["questions"])
    
    @pytest.mark.asyncio
    async def test_rerank_candidates(self, forward_reasoning):
        """Test candidate reranking."""
        mock_futures = [
            {
                "id": "S-A",
                "name": "Direct Path",
                "description": "Straightforward approach",
                "questions": [
                    Question(
                        id="QA1",
                        text="What are our hiring goals?",
                        level=1,
                        cognitive_move=CognitiveMove.DEFINE,
                        builds_on=[],
                        delta_nuance="Defines hiring goals",
                        expected_info_gain=0.8,
                        confidence=0.7
                    )
                ],
                "template": {"focus": "efficiency"}
            },
            {
                "id": "S-B", 
                "name": "Exploratory",
                "description": "Comprehensive exploration",
                "questions": [
                    Question(
                        id="QB1",
                        text="What are all possible approaches?",
                        level=1,
                        cognitive_move=CognitiveMove.SCOPE,
                        builds_on=[],
                        delta_nuance="Scopes approaches",
                        expected_info_gain=0.6,
                        confidence=0.5
                    )
                ],
                "template": {"focus": "comprehensiveness"}
            }
        ]
        
        ranked = await forward_reasoning.rerank_candidates(mock_futures)
        
        # Should return same number of futures
        assert len(ranked) == len(mock_futures)
        
        # Should add lane scores
        for future in ranked:
            assert "lane_score" in future
            assert 0.0 <= future["lane_score"] <= 1.0
        
        # Should be sorted by lane score (descending)
        scores = [f["lane_score"] for f in ranked]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_stitch_scenarios(self, forward_reasoning):
        """Test scenario stitching."""
        mock_candidates = [
            {
                "id": "S-A",
                "name": "Direct Path", 
                "description": "Straightforward approach",
                "questions": [
                    Question(
                        id="QA1",
                        text="What are our goals?",
                        level=1,
                        cognitive_move=CognitiveMove.DEFINE,
                        builds_on=[],
                        delta_nuance="Defines goals",
                        expected_info_gain=0.8,
                        confidence=0.7
                    ),
                    Question(
                        id="QA2", 
                        text="How do we measure progress?",
                        level=2,
                        cognitive_move=CognitiveMove.QUANTIFY,
                        builds_on=[],  # Will be filled by stitching
                        delta_nuance="Adds metrics",
                        expected_info_gain=0.6,
                        confidence=0.5
                    )
                ]
            }
        ]
        
        lanes = await forward_reasoning.stitch_scenarios(mock_candidates)
        
        # Should return Lane objects
        assert len(lanes) == 1
        assert isinstance(lanes[0], Lane)
        
        lane = lanes[0]
        assert lane.id == "S-A"
        assert lane.name == "Direct Path"
        assert len(lane.lane) == 2
        
        # Should have proper builds_on relationships
        level_2_question = next(q for q in lane.lane if q.level == 2)
        assert len(level_2_question.builds_on) > 0
    
    @pytest.mark.asyncio
    async def test_expand_around_node(self, forward_reasoning, sample_budgets):
        """Test node expansion."""
        node = Question(
            id="Q1",
            text="What are our hiring goals?",
            level=1,
            cognitive_move=CognitiveMove.DEFINE,
            builds_on=[],
            delta_nuance="Defines hiring goals",
            expected_info_gain=0.8,
            confidence=0.7
        )
        
        user_answer = "We want to hire 10 senior engineers"
        
        expanded = await forward_reasoning.expand_around_node(
            node, user_answer, sample_budgets
        )
        
        # Should generate follow-up questions
        assert len(expanded) > 0
        assert len(expanded) <= sample_budgets.beam_width
        
        # All should be Question objects
        assert all(isinstance(q, Question) for q in expanded)
        
        # Should build on the original node
        for question in expanded:
            assert node.id in question.builds_on
            assert question.level > node.level
    
    @pytest.mark.asyncio
    async def test_continue_from_node(self, forward_reasoning, sample_budgets):
        """Test node continuation."""
        node = Question(
            id="Q2",
            text="How do we scope the hiring process?",
            level=2,
            cognitive_move=CognitiveMove.SCOPE,
            builds_on=["Q1"],
            delta_nuance="Scopes hiring process",
            expected_info_gain=0.6,
            confidence=0.5
        )
        
        continued = await forward_reasoning.continue_from_node(node, sample_budgets)
        
        # Should generate continuation questions
        assert len(continued) <= sample_budgets.depth_fwd
        
        # Should form a chain
        if continued:
            assert continued[0].level == node.level + 1
            assert node.id in continued[0].builds_on
            
            # Each subsequent question should build on previous
            for i in range(1, len(continued)):
                assert continued[i].level == continued[i-1].level + 1
                assert continued[i-1].id in continued[i].builds_on
    
    def test_choose_cognitive_move_for_level(self, forward_reasoning):
        """Test cognitive move selection."""
        emphasis_moves = [CognitiveMove.COMPARE, CognitiveMove.SIMULATE]
        
        # Test with emphasis
        with patch('random.random', return_value=0.5):  # 50% < 70% threshold
            move = forward_reasoning._choose_cognitive_move_for_level(3, emphasis_moves)
            assert move in emphasis_moves
        
        # Test without emphasis (random > threshold)
        with patch('random.random', return_value=0.8):  # 80% > 70% threshold
            move = forward_reasoning._choose_cognitive_move_for_level(3, emphasis_moves)
            # Should use default for level 3
            assert move == CognitiveMove.QUANTIFY
    
    def test_generate_question_text(self, forward_reasoning):
        """Test question text generation."""
        current_question = "How do we improve customer satisfaction?"
        template = {"focus": "efficiency", "name": "Direct Path"}
        
        question_text = forward_reasoning._generate_question_text(
            current_question, template, CognitiveMove.DEFINE, 1, None
        )
        
        assert isinstance(question_text, str)
        assert len(question_text) > 0
        assert "efficiency" in question_text.lower()
    
    def test_extract_key_terms(self, forward_reasoning):
        """Test key term extraction."""
        text = "How should we optimize our Customer Success metrics?"
        
        key_terms = forward_reasoning._extract_key_terms(text)
        
        assert isinstance(key_terms, list)
        assert len(key_terms) > 0
        # Should extract "Customer" and "Success"
        assert any("customer" in term.lower() for term in key_terms)
    
    def test_generate_delta_nuance(self, forward_reasoning):
        """Test delta nuance generation."""
        template = {"focus": "efficiency"}
        
        nuance = forward_reasoning._generate_delta_nuance(CognitiveMove.DEFINE, template)
        
        assert isinstance(nuance, str)
        assert "efficiency" in nuance.lower()
        assert "define" in nuance.lower() or "definition" in nuance.lower()
    
    @pytest.mark.asyncio
    async def test_stitch_lane_questions(self, forward_reasoning):
        """Test lane question stitching."""
        questions = [
            Question(
                id="QA2",
                text="Second question",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=[],
                delta_nuance="Second",
                expected_info_gain=0.6,
                confidence=0.5
            ),
            Question(
                id="QA1",
                text="First question", 
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="First",
                expected_info_gain=0.8,
                confidence=0.7
            )
        ]
        
        stitched = await forward_reasoning._stitch_lane_questions(questions)
        
        # Should sort by level
        assert stitched[0].level < stitched[1].level
        
        # Level 1 should have no dependencies
        assert stitched[0].builds_on == []
        
        # Level 2 should have dependencies
        assert len(stitched[1].builds_on) > 0
    
    def test_add_triggers_to_questions(self, forward_reasoning):
        """Test trigger addition."""
        questions = [
            Question(
                id="Q1",
                text="How do we measure success?",
                level=1,
                cognitive_move=CognitiveMove.QUANTIFY,
                builds_on=[],
                delta_nuance="Adds metrics",
                expected_info_gain=0.7,
                confidence=0.6
            ),
            Question(
                id="Q2",
                text="What scenarios should we model?",
                level=2,
                cognitive_move=CognitiveMove.SIMULATE,
                builds_on=["Q1"],
                delta_nuance="Models scenarios",
                expected_info_gain=0.5,
                confidence=0.4
            )
        ]
        
        forward_reasoning._add_triggers_to_questions(questions)
        
        # QUANTIFY question should have metric trigger
        quantify_q = next(q for q in questions if q.cognitive_move == CognitiveMove.QUANTIFY)
        assert len(quantify_q.triggers) > 0
        assert quantify_q.triggers[0].type == TriggerType.METRIC
        
        # SIMULATE question should have event trigger
        simulate_q = next(q for q in questions if q.cognitive_move == CognitiveMove.SIMULATE)
        assert len(simulate_q.triggers) > 0
        assert simulate_q.triggers[0].type == TriggerType.EVENT
    
    def test_mark_natural_endings(self, forward_reasoning):
        """Test natural ending detection."""
        questions = [
            Question(
                id="Q1",
                text="What should we do next?",
                level=5,
                cognitive_move=CognitiveMove.COMMIT,  # Terminal move
                builds_on=["Q4"],
                delta_nuance="Commits to action",
                expected_info_gain=0.2,  # Low info gain
                confidence=0.3
            ),
            Question(
                id="Q2",
                text="How do we define success?",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Defines success",
                expected_info_gain=0.8,  # High info gain
                confidence=0.7
            )
        ]
        
        forward_reasoning._mark_natural_endings(questions)
        
        # COMMIT question should be marked as natural end
        commit_q = next(q for q in questions if q.cognitive_move == CognitiveMove.COMMIT)
        assert commit_q.natural_end is True
        
        # DEFINE question should not be marked as natural end
        define_q = next(q for q in questions if q.cognitive_move == CognitiveMove.DEFINE)
        assert define_q.natural_end is False
    
    def test_add_cross_links(self, forward_reasoning):
        """Test cross-link addition between lanes."""
        lane1 = Lane(
            id="S-A",
            name="Lane A",
            description="First lane",
            lane=[
                Question(
                    id="QA1",
                    text="How do we compare options?",
                    level=2,
                    cognitive_move=CognitiveMove.COMPARE,
                    builds_on=["QA0"],
                    delta_nuance="Compares options",
                    expected_info_gain=0.6,
                    confidence=0.5
                )
            ]
        )
        
        lane2 = Lane(
            id="S-B", 
            name="Lane B",
            description="Second lane",
            lane=[
                Question(
                    id="QB1",
                    text="What alternatives should we compare?",
                    level=2,
                    cognitive_move=CognitiveMove.COMPARE,
                    builds_on=["QB0"],
                    delta_nuance="Compares alternatives",
                    expected_info_gain=0.7,
                    confidence=0.6
                )
            ]
        )
        
        lanes = [lane1, lane2]
        forward_reasoning._add_cross_links(lanes)
        
        # Should add cross-links between similar questions
        # At least one lane should have cross-links
        total_cross_links = sum(len(lane.cross_links) for lane in lanes)
        assert total_cross_links > 0
    
    def test_calculate_progression_quality(self, forward_reasoning):
        """Test progression quality calculation."""
        questions = [
            Question(
                id="Q1",
                text="First question",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="First",
                expected_info_gain=0.8,
                confidence=0.7
            ),
            Question(
                id="Q2",
                text="Second question",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,  # Valid progression from DEFINE
                builds_on=["Q1"],
                delta_nuance="Second",
                expected_info_gain=0.6,
                confidence=0.5
            )
        ]
        
        quality = forward_reasoning._calculate_progression_quality(questions)
        
        # Should be high quality (valid progression)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be good quality
    
    @pytest.mark.asyncio
    async def test_lane_template_coverage(self, forward_reasoning, sample_budgets):
        """Test that all lane templates are used."""
        current_question = "How should we approach this problem?"
        
        # Generate multiple sets of futures to test template variety
        all_template_names = set()
        
        for _ in range(3):  # Run multiple times to get variety
            futures = await forward_reasoning.enumerate_futures(
                current_question, None, sample_budgets
            )
            
            for future in futures:
                all_template_names.add(future["name"])
        
        # Should use different templates
        assert len(all_template_names) > 1
        
        # Should include expected template types
        expected_types = ["Direct", "Exploratory", "Risk", "Innovative", "Resource"]
        assert any(expected in name for expected in expected_types for name in all_template_names)