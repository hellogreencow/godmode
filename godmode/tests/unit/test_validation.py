"""Test invariant validation system."""

import pytest
from godmode.core.validation import InvariantValidator
from godmode.models.core import Question, Lane, Thread, CognitiveMove, ThreadStatus
from godmode.models.responses import GraphUpdate, Meta


class TestInvariantValidator:
    """Test InvariantValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create InvariantValidator instance."""
        return InvariantValidator()
    
    @pytest.fixture
    def valid_questions(self):
        """Create valid question set."""
        return [
            Question(
                id="Q1",
                text="What is our goal?",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Defines goal",
                expected_info_gain=0.8,
                confidence=0.7
            ),
            Question(
                id="Q2",
                text="What are the constraints?",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=["Q1"],
                delta_nuance="Adds constraints",
                expected_info_gain=0.6,
                confidence=0.5
            ),
            Question(
                id="Q3",
                text="How do we measure progress?",
                level=3,
                cognitive_move=CognitiveMove.QUANTIFY,
                builds_on=["Q2"],
                delta_nuance="Adds metrics",
                expected_info_gain=0.5,
                confidence=0.4
            )
        ]
    
    def test_validate_valid_questions(self, validator, valid_questions):
        """Test validation of valid question set."""
        is_valid, errors = validator.validate_questions(valid_questions)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_dag_structure_no_cycles(self, validator, valid_questions):
        """Test DAG validation with no cycles."""
        validator._validate_dag_structure(valid_questions)
        
        # Should have no cycle errors
        cycle_errors = [e for e in validator.validation_errors if "cycle" in e.lower()]
        assert len(cycle_errors) == 0
    
    def test_validate_dag_structure_with_cycle(self, validator):
        """Test DAG validation with cycle detection."""
        # Create questions with a cycle: Q1 -> Q2 -> Q3 -> Q1
        cyclic_questions = [
            Question(
                id="Q1",
                text="First question",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=["Q3"],  # Creates cycle
                delta_nuance="First",
                expected_info_gain=0.5,
                confidence=0.5
            ),
            Question(
                id="Q2",
                text="Second question",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=["Q1"],
                delta_nuance="Second",
                expected_info_gain=0.5,
                confidence=0.5
            ),
            Question(
                id="Q3",
                text="Third question",
                level=3,
                cognitive_move=CognitiveMove.QUANTIFY,
                builds_on=["Q2"],
                delta_nuance="Third",
                expected_info_gain=0.5,
                confidence=0.5
            )
        ]
        
        validator._validate_dag_structure(cyclic_questions)
        
        # Should detect cycle
        cycle_errors = [e for e in validator.validation_errors if "cycle" in e.lower()]
        assert len(cycle_errors) > 0
    
    def test_validate_level_progression_valid(self, validator, valid_questions):
        """Test level progression validation with valid progression."""
        validator._validate_level_progression(valid_questions)
        
        # Should have no level progression errors
        level_errors = [e for e in validator.validation_errors if "level" in e.lower()]
        assert len(level_errors) == 0
    
    def test_validate_level_progression_invalid(self, validator):
        """Test level progression validation with invalid progression."""
        invalid_questions = [
            Question(
                id="Q1",
                text="First question",
                level=2,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="First",
                expected_info_gain=0.5,
                confidence=0.5
            ),
            Question(
                id="Q2",
                text="Second question",
                level=1,  # Invalid: level <= parent level
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=["Q1"],
                delta_nuance="Second",
                expected_info_gain=0.5,
                confidence=0.5
            )
        ]
        
        validator._validate_level_progression(invalid_questions)
        
        # Should detect level progression violation
        level_errors = [e for e in validator.validation_errors if "level" in e.lower()]
        assert len(level_errors) > 0
    
    def test_validate_no_duplicates_valid(self, validator):
        """Test duplicate validation with no duplicates."""
        priors = [
            Question(
                id="QP1",
                text="What is our goal?",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Defines goal",
                expected_info_gain=0.8,
                confidence=0.7
            ),
            Question(
                id="QP2",
                text="What are the constraints?",
                level=1,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=[],
                delta_nuance="Adds constraints",
                expected_info_gain=0.6,
                confidence=0.5
            )
        ]
        
        scenarios = [
            Lane(
                id="S-A",
                name="Direct Path",
                description="Straightforward approach",
                lane=[
                    Question(
                        id="QA1",
                        text="How do we proceed?",
                        level=1,
                        cognitive_move=CognitiveMove.DEFINE,
                        builds_on=[],
                        delta_nuance="Defines approach",
                        expected_info_gain=0.7,
                        confidence=0.6
                    )
                ]
            )
        ]
        
        graph_update = GraphUpdate(
            current_question="Test question",
            priors=priors,
            scenarios=scenarios,
            threads=[],
            meta=Meta(budgets_used={})
        )
        
        validator._validate_no_duplicates(graph_update)
        
        # Should have no duplicate errors
        duplicate_errors = [e for e in validator.validation_errors if "duplicate" in e.lower()]
        assert len(duplicate_errors) == 0
    
    def test_validate_no_duplicates_with_duplicates(self, validator):
        """Test duplicate validation with duplicates."""
        priors = [
            Question(
                id="QP1",
                text="What is our goal?",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Defines goal",
                expected_info_gain=0.8,
                confidence=0.7
            ),
            Question(
                id="QP2",
                text="What is our goal?",  # Exact duplicate
                level=1,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=[],
                delta_nuance="Also defines goal",
                expected_info_gain=0.6,
                confidence=0.5
            )
        ]
        
        graph_update = GraphUpdate(
            current_question="Test question",
            priors=priors,
            scenarios=[],
            threads=[],
            meta=Meta(budgets_used={})
        )
        
        validator._validate_no_duplicates(graph_update)
        
        # Should detect duplicate
        duplicate_errors = [e for e in validator.validation_errors if "duplicate" in e.lower()]
        assert len(duplicate_errors) > 0
    
    def test_validate_scenario_constraints_valid(self, validator):
        """Test scenario constraints validation with valid scenarios."""
        scenarios = [
            Lane(id="S-A", name="Lane A", description="First", lane=[
                Question(
                    id="QA1", text="Question A1", level=1, cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[], delta_nuance="A1", expected_info_gain=0.5, confidence=0.5
                )
            ]),
            Lane(id="S-B", name="Lane B", description="Second", lane=[
                Question(
                    id="QB1", text="Question B1", level=1, cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[], delta_nuance="B1", expected_info_gain=0.5, confidence=0.5
                )
            ]),
            Lane(id="S-C", name="Lane C", description="Third", lane=[
                Question(
                    id="QC1", text="Question C1", level=1, cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[], delta_nuance="C1", expected_info_gain=0.5, confidence=0.5
                )
            ])
        ]
        
        validator._validate_scenario_constraints(scenarios)
        
        # Should have no scenario errors
        scenario_errors = [e for e in validator.validation_errors if "scenario" in e.lower() or "lane" in e.lower()]
        assert len(scenario_errors) == 0
    
    def test_validate_scenario_constraints_wrong_count(self, validator):
        """Test scenario constraints with wrong number of scenarios."""
        # Only 2 scenarios (should be 3-5)
        scenarios = [
            Lane(id="S-A", name="Lane A", description="First", lane=[
                Question(
                    id="QA1", text="Question A1", level=1, cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[], delta_nuance="A1", expected_info_gain=0.5, confidence=0.5
                )
            ]),
            Lane(id="S-B", name="Lane B", description="Second", lane=[
                Question(
                    id="QB1", text="Question B1", level=1, cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[], delta_nuance="B1", expected_info_gain=0.5, confidence=0.5
                )
            ])
        ]
        
        validator._validate_scenario_constraints(scenarios)
        
        # Should detect wrong scenario count
        scenario_errors = [e for e in validator.validation_errors if "3-5 scenario" in e]
        assert len(scenario_errors) > 0
    
    def test_validate_scenario_constraints_empty_lane(self, validator):
        """Test scenario constraints with empty lane."""
        scenarios = [
            Lane(id="S-A", name="Lane A", description="First", lane=[]),  # Empty lane
            Lane(id="S-B", name="Lane B", description="Second", lane=[
                Question(
                    id="QB1", text="Question B1", level=1, cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[], delta_nuance="B1", expected_info_gain=0.5, confidence=0.5
                )
            ]),
            Lane(id="S-C", name="Lane C", description="Third", lane=[
                Question(
                    id="QC1", text="Question C1", level=1, cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[], delta_nuance="C1", expected_info_gain=0.5, confidence=0.5
                )
            ])
        ]
        
        validator._validate_scenario_constraints(scenarios)
        
        # Should detect empty lane
        empty_errors = [e for e in validator.validation_errors if "no questions" in e]
        assert len(empty_errors) > 0
    
    def test_validate_thread_consistency_valid(self, validator, valid_questions):
        """Test thread consistency validation with valid threads."""
        threads = [
            Thread(
                thread_id="T1",
                origin_node_id="Q1",
                path=["Q1", "Q2", "Q3"],
                status=ThreadStatus.ACTIVE,
                summary="Valid thread path"
            )
        ]
        
        validator._validate_thread_consistency(threads, valid_questions)
        
        # Should have no thread errors
        thread_errors = [e for e in validator.validation_errors if "thread" in e.lower()]
        assert len(thread_errors) == 0
    
    def test_validate_thread_consistency_nonexistent_node(self, validator, valid_questions):
        """Test thread consistency with non-existent node."""
        threads = [
            Thread(
                thread_id="T1",
                origin_node_id="Q1",
                path=["Q1", "Q2", "Q999"],  # Q999 doesn't exist
                status=ThreadStatus.ACTIVE,
                summary="Invalid thread path"
            )
        ]
        
        validator._validate_thread_consistency(threads, valid_questions)
        
        # Should detect non-existent node
        thread_errors = [e for e in validator.validation_errors if "non-existent node" in e]
        assert len(thread_errors) > 0
    
    def test_validate_thread_consistency_origin_mismatch(self, validator, valid_questions):
        """Test thread consistency with origin mismatch."""
        threads = [
            Thread(
                thread_id="T1",
                origin_node_id="Q1",
                path=["Q2", "Q3"],  # Path doesn't start with origin
                status=ThreadStatus.ACTIVE,
                summary="Mismatched origin"
            )
        ]
        
        validator._validate_thread_consistency(threads, valid_questions)
        
        # Should detect origin mismatch
        origin_errors = [e for e in validator.validation_errors if "origin" in e and "doesn't match" in e]
        assert len(origin_errors) > 0
    
    def test_validate_lane(self, validator):
        """Test lane validation."""
        valid_lane = Lane(
            id="S-A",
            name="Test Lane",
            description="Test description",
            lane=[
                Question(
                    id="QA1",
                    text="First question",
                    level=1,
                    cognitive_move=CognitiveMove.DEFINE,
                    builds_on=[],
                    delta_nuance="First",
                    expected_info_gain=0.8,
                    confidence=0.7
                ),
                Question(
                    id="QA2",
                    text="Second question",
                    level=2,
                    cognitive_move=CognitiveMove.SCOPE,
                    builds_on=["QA1"],
                    delta_nuance="Second",
                    expected_info_gain=0.6,
                    confidence=0.5
                )
            ]
        )
        
        is_valid, errors = validator.validate_lane(valid_lane)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_natural_endings(self, validator):
        """Test natural ending validation."""
        questions = [
            Question(
                id="Q1",
                text="Low level natural end",
                level=1,  # Too low for natural end
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Low level end",
                expected_info_gain=0.8,  # High info gain (inconsistent with natural end)
                confidence=0.7,
                natural_end=True
            ),
            Question(
                id="Q2",
                text="Valid natural end",
                level=5,  # High level
                cognitive_move=CognitiveMove.COMMIT,
                builds_on=["Q1"],
                delta_nuance="Valid end",
                expected_info_gain=0.2,  # Low info gain (consistent with natural end)
                confidence=0.3,
                natural_end=True
            )
        ]
        
        validator._validate_natural_endings(questions)
        
        # Should detect issues with first question
        natural_end_errors = [e for e in validator.validation_errors if "natural_end" in e]
        assert len(natural_end_errors) >= 1  # At least one error for Q1
    
    def test_calculate_text_similarity(self, validator):
        """Test text similarity calculation."""
        text1 = "What is our primary goal for customer success?"
        text2 = "What is our main objective for customer satisfaction?"
        text3 = "How do we implement the payment system?"
        
        # Similar texts
        similarity1 = validator._calculate_text_similarity(text1, text2)
        assert 0.0 < similarity1 < 1.0
        assert similarity1 > 0.3  # Should be fairly similar
        
        # Dissimilar texts
        similarity2 = validator._calculate_text_similarity(text1, text3)
        assert similarity2 < 0.3  # Should be less similar
        
        # Identical texts
        similarity3 = validator._calculate_text_similarity(text1, text1)
        assert similarity3 == 1.0
    
    def test_get_validation_summary(self, validator):
        """Test validation summary generation."""
        # Add some mock errors
        validator.validation_errors = [
            "Cycle detected involving node Q1",
            "Level progression violation: Q2 must have level > Q1",
            "Duplicate question text found in priors at level 1",
            "Invalid cognitive progression: Q1 -> Q2",
            "Must have 3-5 scenario lanes, got 2",
            "Thread T1 references non-existent node Q999",
            "Some other error"
        ]
        
        summary = validator.get_validation_summary()
        
        assert summary["total_errors"] == 7
        assert summary["error_breakdown"]["dag_violations"] == 1
        assert summary["error_breakdown"]["level_violations"] == 1
        assert summary["error_breakdown"]["duplicate_violations"] == 1
        assert summary["error_breakdown"]["cognitive_violations"] == 1
        assert summary["error_breakdown"]["scenario_violations"] == 1
        assert summary["error_breakdown"]["thread_violations"] == 1
        assert summary["error_breakdown"]["other_violations"] == 1
        assert summary["is_valid"] is False
    
    def test_empty_input_handling(self, validator):
        """Test handling of empty inputs."""
        # Empty questions
        is_valid, errors = validator.validate_questions([])
        assert is_valid is True
        assert len(errors) == 0
        
        # Empty graph update
        empty_graph = GraphUpdate(
            current_question="",
            priors=[],
            scenarios=[],
            threads=[],
            meta=Meta(budgets_used={})
        )
        
        is_valid, errors = validator.validate_graph_update(empty_graph)
        # Should fail due to scenario count constraint (0 < 3)
        assert is_valid is False
        assert any("3-5 scenario" in error for error in errors)