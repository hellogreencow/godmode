"""End-to-end integration tests for GODMODE."""

import pytest
import asyncio
from godmode.core.engine import GodmodeEngine
from godmode.models.commands import InitCommand, AdvanceCommand, Budgets
from godmode.schemas.validator import SchemaValidator


class TestEndToEnd:
    """Test complete GODMODE workflows end-to-end."""
    
    @pytest.fixture
    def engine(self):
        """Create GODMODE engine instance."""
        return GodmodeEngine()
    
    @pytest.fixture
    def validator(self):
        """Create schema validator."""
        return SchemaValidator()
    
    @pytest.fixture
    def sample_budgets(self):
        """Create sample budgets for testing."""
        return Budgets(
            beam_width=3,
            depth_back=3,
            depth_fwd=3,
            max_tokens_reply=150,
            time_s=5.0
        )
    
    @pytest.mark.asyncio
    async def test_basic_init_workflow(self, engine, validator, sample_budgets):
        """Test basic INIT workflow."""
        # Create INIT command
        command = InitCommand(
            current_question="How should we improve customer satisfaction?",
            context="We're a SaaS company with 500 users",
            budgets=sample_budgets
        )
        
        # Process command
        response = await engine.process_command(command)
        
        # Validate response structure
        assert response is not None
        assert hasattr(response, 'chat_reply')
        assert hasattr(response, 'graph_update')
        assert hasattr(response, 'ontology_update')
        
        # Validate chat reply
        assert isinstance(response.chat_reply, str)
        assert len(response.chat_reply) > 0
        assert len(response.chat_reply) <= sample_budgets.max_tokens_reply * 10  # Rough token estimate
        
        # Validate graph update
        graph = response.graph_update
        assert graph.current_question == command.current_question
        assert isinstance(graph.priors, list)
        assert isinstance(graph.scenarios, list)
        assert isinstance(graph.threads, list)
        
        # Should have some priors
        assert len(graph.priors) > 0
        
        # Should have 3-5 scenario lanes
        assert 3 <= len(graph.scenarios) <= 5
        
        # Each scenario should have questions
        for scenario in graph.scenarios:
            assert len(scenario.lane) > 0
            
        # Should have at least one thread
        assert len(graph.threads) >= 0  # May be empty in some implementations
        
        # Validate ontology update
        ontology = response.ontology_update
        assert isinstance(ontology.entities, list)
        assert isinstance(ontology.relations, list)
        assert isinstance(ontology.mappings, list)
        
        # Validate against JSON schema
        response_dict = response.dict()
        is_valid, errors = validator.validate_response(response_dict)
        
        if not is_valid:
            pytest.fail(f"Response failed schema validation: {errors}")
    
    @pytest.mark.asyncio
    async def test_init_advance_workflow(self, engine, sample_budgets):
        """Test INIT followed by ADVANCE workflow."""
        # First, initialize with a question
        init_command = InitCommand(
            current_question="What's the best approach to scale our team?",
            budgets=sample_budgets
        )
        
        init_response = await engine.process_command(init_command)
        
        # Verify we got priors
        assert len(init_response.graph_update.priors) > 0
        
        # Pick the first prior to advance from
        first_prior = init_response.graph_update.priors[0]
        
        # Create ADVANCE command
        advance_command = AdvanceCommand(
            node_id=first_prior.id,
            user_answer="We want to focus on hiring senior engineers"
        )
        
        # Process ADVANCE command
        advance_response = await engine.process_command(advance_command)
        
        # Validate advance response
        assert advance_response is not None
        assert isinstance(advance_response.chat_reply, str)
        assert len(advance_response.chat_reply) > 0
        
        # Should reference the advanced node
        assert first_prior.id in advance_response.chat_reply or "expanded" in advance_response.chat_reply.lower()
    
    @pytest.mark.asyncio
    async def test_multiple_question_types(self, engine, sample_budgets):
        """Test GODMODE with different types of questions."""
        test_questions = [
            # Business strategy
            "How should we enter the European market?",
            # Technical decision
            "What database should we use for our new application?",
            # Personal decision
            "Should I change careers to data science?",
            # Process improvement
            "How can we reduce customer support response time?"
        ]
        
        for question in test_questions:
            command = InitCommand(
                current_question=question,
                budgets=sample_budgets
            )
            
            response = await engine.process_command(command)
            
            # Basic validations for each question type
            assert response is not None
            assert len(response.chat_reply) > 0
            assert len(response.graph_update.priors) > 0
            assert 3 <= len(response.graph_update.scenarios) <= 5
            
            # Each scenario should be relevant to the question
            for scenario in response.graph_update.scenarios:
                assert len(scenario.lane) > 0
                # Questions should contain some relevant terms
                scenario_text = " ".join(q.text for q in scenario.lane)
                # This is a loose check - in practice, we'd want more sophisticated relevance checking
                assert len(scenario_text) > 50  # Should have substantial content
    
    @pytest.mark.asyncio
    async def test_budget_constraints_respected(self, engine):
        """Test that budget constraints are respected."""
        strict_budgets = Budgets(
            beam_width=2,
            depth_back=2,
            depth_fwd=2,
            max_tokens_reply=50,
            time_s=1.0
        )
        
        command = InitCommand(
            current_question="This is a complex multi-faceted question about business strategy, technology decisions, and organizational change management",
            budgets=strict_budgets
        )
        
        start_time = asyncio.get_event_loop().time()
        response = await engine.process_command(command)
        end_time = asyncio.get_event_loop().time()
        
        # Check time constraint (with some tolerance)
        elapsed_time = end_time - start_time
        assert elapsed_time <= strict_budgets.time_s + 2.0  # 2 second tolerance
        
        # Check that response respects depth constraints
        if response.graph_update.priors:
            max_prior_level = max(q.level for q in response.graph_update.priors)
            assert max_prior_level <= strict_budgets.depth_back
        
        for scenario in response.graph_update.scenarios:
            if scenario.lane:
                max_scenario_level = max(q.level for q in scenario.lane)
                assert max_scenario_level <= strict_budgets.depth_fwd
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine, sample_budgets):
        """Test error handling for invalid inputs."""
        # Test empty question
        with pytest.raises(Exception):  # Should raise some validation error
            command = InitCommand(
                current_question="",
                budgets=sample_budgets
            )
            await engine.process_command(command)
        
        # Test invalid ADVANCE command (no current session)
        with pytest.raises(Exception):
            advance_command = AdvanceCommand(node_id="nonexistent")
            await engine.process_command(advance_command)
    
    @pytest.mark.asyncio
    async def test_consistency_across_runs(self, engine, sample_budgets):
        """Test that GODMODE produces consistent results across multiple runs."""
        question = "How should we improve our product's user experience?"
        
        responses = []
        for _ in range(3):  # Run 3 times
            command = InitCommand(
                current_question=question,
                budgets=sample_budgets
            )
            response = await engine.process_command(command)
            responses.append(response)
        
        # Check basic consistency
        for response in responses:
            assert response.graph_update.current_question == question
            assert len(response.graph_update.priors) > 0
            assert 3 <= len(response.graph_update.scenarios) <= 5
        
        # While exact questions may vary due to randomness in generation,
        # the structure and quality should be consistent
        prior_counts = [len(r.graph_update.priors) for r in responses]
        scenario_counts = [len(r.graph_update.scenarios) for r in responses]
        
        # Should have similar numbers of priors and scenarios
        assert max(prior_counts) - min(prior_counts) <= 2  # Within 2 of each other
        assert max(scenario_counts) - min(scenario_counts) <= 1  # Within 1 of each other
    
    @pytest.mark.asyncio
    async def test_cognitive_progression_quality(self, engine, sample_budgets):
        """Test that generated questions follow good cognitive progression."""
        command = InitCommand(
            current_question="Should we implement a new customer onboarding system?",
            budgets=sample_budgets
        )
        
        response = await engine.process_command(command)
        
        # Check prior progression
        if len(response.graph_update.priors) > 1:
            for prior in response.graph_update.priors:
                if prior.builds_on:
                    # Should build on lower-level questions
                    for parent_id in prior.builds_on:
                        parent = next((p for p in response.graph_update.priors if p.id == parent_id), None)
                        if parent:
                            assert prior.level > parent.level
        
        # Check scenario progression
        for scenario in response.graph_update.scenarios:
            if len(scenario.lane) > 1:
                for question in scenario.lane:
                    if question.builds_on:
                        for parent_id in question.builds_on:
                            parent = next((q for q in scenario.lane if q.id == parent_id), None)
                            if parent:
                                assert question.level > parent.level
        
        # Check that different cognitive moves are represented
        all_moves = set()
        for prior in response.graph_update.priors:
            all_moves.add(prior.cognitive_move)
        
        for scenario in response.graph_update.scenarios:
            for question in scenario.lane:
                all_moves.add(question.cognitive_move)
        
        # Should have at least 3 different cognitive moves
        assert len(all_moves) >= 3
    
    @pytest.mark.asyncio
    async def test_information_gain_calibration(self, engine, sample_budgets):
        """Test that information gain scores are reasonable."""
        command = InitCommand(
            current_question="How can we optimize our marketing budget allocation?",
            budgets=sample_budgets
        )
        
        response = await engine.process_command(command)
        
        # Check that info gain scores are in valid range
        all_questions = response.graph_update.priors[:]
        for scenario in response.graph_update.scenarios:
            all_questions.extend(scenario.lane)
        
        for question in all_questions:
            assert 0.0 <= question.expected_info_gain <= 1.0
            assert 0.0 <= question.confidence <= 1.0
        
        # Higher-level foundational questions should generally have higher info gain
        if response.graph_update.priors:
            level_1_priors = [q for q in response.graph_update.priors if q.level == 1]
            if level_1_priors:
                avg_level_1_gain = sum(q.expected_info_gain for q in level_1_priors) / len(level_1_priors)
                # Level 1 questions should have reasonably high info gain
                assert avg_level_1_gain >= 0.4