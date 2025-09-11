"""Test core data models."""

import pytest
from pydantic import ValidationError
from datetime import datetime

from godmode.models.core import (
    Question, Lane, Thread, Entity, Relation, Mapping,
    CognitiveMove, EntityType, ThreadStatus, Trigger, TriggerType,
    CrossLink, TemporalInfo, Evidence
)


class TestQuestion:
    """Test Question model."""
    
    def test_valid_question_creation(self):
        """Test creating a valid question."""
        question = Question(
            id="Q1",
            text="What is the primary goal?",
            level=1,
            cognitive_move=CognitiveMove.DEFINE,
            builds_on=[],
            delta_nuance="Establishes primary goal definition",
            expected_info_gain=0.8,
            confidence=0.7,
            tags=["goal", "definition"]
        )
        
        assert question.id == "Q1"
        assert question.level == 1
        assert question.cognitive_move == CognitiveMove.DEFINE
        assert question.builds_on == []
        assert question.expected_info_gain == 0.8
        assert question.confidence == 0.7
        assert not question.natural_end
        assert question.triggers == []
    
    def test_question_validation_level_too_low(self):
        """Test that level must be >= 1."""
        with pytest.raises(ValidationError):
            Question(
                id="Q1",
                text="Test question",
                level=0,  # Invalid
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Test nuance",
                expected_info_gain=0.5,
                confidence=0.5
            )
    
    def test_question_validation_info_gain_out_of_range(self):
        """Test that expected_info_gain must be in [0,1]."""
        with pytest.raises(ValidationError):
            Question(
                id="Q1",
                text="Test question",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Test nuance",
                expected_info_gain=1.5,  # Invalid
                confidence=0.5
            )
    
    def test_question_validation_confidence_out_of_range(self):
        """Test that confidence must be in [0,1]."""
        with pytest.raises(ValidationError):
            Question(
                id="Q1",
                text="Test question",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Test nuance",
                expected_info_gain=0.5,
                confidence=-0.1  # Invalid
            )
    
    def test_question_builds_on_validation_level_1(self):
        """Test that level 1 questions can have empty builds_on."""
        question = Question(
            id="Q1",
            text="Test question",
            level=1,
            cognitive_move=CognitiveMove.DEFINE,
            builds_on=[],  # Valid for level 1
            delta_nuance="Test nuance",
            expected_info_gain=0.5,
            confidence=0.5
        )
        assert question.builds_on == []
    
    def test_question_builds_on_validation_higher_level(self):
        """Test that higher level questions must have builds_on."""
        with pytest.raises(ValidationError):
            Question(
                id="Q2",
                text="Test question",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=[],  # Invalid for level > 1
                delta_nuance="Test nuance",
                expected_info_gain=0.5,
                confidence=0.5
            )
    
    def test_question_with_triggers(self):
        """Test question with triggers."""
        trigger = Trigger(type=TriggerType.METRIC, detail="When metrics are defined")
        
        question = Question(
            id="Q1",
            text="How do we measure success?",
            level=1,
            cognitive_move=CognitiveMove.QUANTIFY,
            builds_on=[],
            delta_nuance="Adds measurement framework",
            expected_info_gain=0.7,
            confidence=0.6,
            triggers=[trigger]
        )
        
        assert len(question.triggers) == 1
        assert question.triggers[0].type == TriggerType.METRIC


class TestLane:
    """Test Lane model."""
    
    def test_valid_lane_creation(self):
        """Test creating a valid lane."""
        questions = [
            Question(
                id="QA1",
                text="What is the goal?",
                level=1,
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="Defines goal",
                expected_info_gain=0.8,
                confidence=0.7
            ),
            Question(
                id="QA2",
                text="What are the constraints?",
                level=2,
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=["QA1"],
                delta_nuance="Adds constraints",
                expected_info_gain=0.6,
                confidence=0.5
            )
        ]
        
        lane = Lane(
            id="S-A",
            name="Direct Path",
            description="Straightforward approach",
            lane=questions
        )
        
        assert lane.id == "S-A"
        assert lane.name == "Direct Path"
        assert len(lane.lane) == 2
        assert lane.cross_links == []
    
    def test_lane_level_progression_validation(self):
        """Test that lane validates level progression."""
        questions = [
            Question(
                id="QA1",
                text="First question",
                level=2,  # Level 2
                cognitive_move=CognitiveMove.DEFINE,
                builds_on=[],
                delta_nuance="First",
                expected_info_gain=0.5,
                confidence=0.5
            ),
            Question(
                id="QA2",
                text="Second question",
                level=1,  # Level 1 - invalid progression
                cognitive_move=CognitiveMove.SCOPE,
                builds_on=["QA1"],
                delta_nuance="Second",
                expected_info_gain=0.5,
                confidence=0.5
            )
        ]
        
        with pytest.raises(ValidationError):
            Lane(
                id="S-A",
                name="Test Lane",
                description="Test",
                lane=questions
            )


class TestThread:
    """Test Thread model."""
    
    def test_valid_thread_creation(self):
        """Test creating a valid thread."""
        thread = Thread(
            thread_id="T1",
            origin_node_id="Q1",
            path=["Q1", "Q2", "Q3"],
            status=ThreadStatus.ACTIVE,
            summary="Thread exploring goal definition"
        )
        
        assert thread.thread_id == "T1"
        assert thread.origin_node_id == "Q1"
        assert thread.path == ["Q1", "Q2", "Q3"]
        assert thread.status == ThreadStatus.ACTIVE
        assert len(thread.summary) <= 280
    
    def test_thread_summary_length_validation(self):
        """Test that thread summary is limited to 280 characters."""
        long_summary = "x" * 300
        
        with pytest.raises(ValidationError):
            Thread(
                thread_id="T1",
                origin_node_id="Q1",
                path=["Q1"],
                status=ThreadStatus.ACTIVE,
                summary=long_summary
            )


class TestEntity:
    """Test Entity model."""
    
    def test_valid_entity_creation(self):
        """Test creating a valid entity."""
        entity = Entity(
            id="E1",
            name="Customer Success",
            type=EntityType.GOAL,
            aliases=["customer satisfaction", "client success"],
            confidence=0.8
        )
        
        assert entity.id == "E1"
        assert entity.name == "Customer Success"
        assert entity.type == EntityType.GOAL
        assert len(entity.aliases) == 2
        assert entity.confidence == 0.8
    
    def test_entity_confidence_validation(self):
        """Test entity confidence validation."""
        with pytest.raises(ValidationError):
            Entity(
                id="E1",
                name="Test Entity",
                type=EntityType.CONCEPT,
                confidence=1.5  # Invalid
            )


class TestRelation:
    """Test Relation model."""
    
    def test_valid_relation_creation(self):
        """Test creating a valid relation."""
        evidence = Evidence(
            source_type="user",
            snippet="Customer success depends on product quality",
            note="From user input"
        )
        
        temporal = TemporalInfo(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31),
            grain="year"
        )
        
        relation = Relation(
            id="R1",
            subj="E1",
            pred="depends_on",
            obj="E2",
            confidence=0.7,
            hypothesis=True,
            evidence=[evidence],
            temporal=temporal
        )
        
        assert relation.id == "R1"
        assert relation.subj == "E1"
        assert relation.pred == "depends_on"
        assert relation.obj == "E2"
        assert relation.confidence == 0.7
        assert relation.hypothesis is True
        assert len(relation.evidence) == 1
        assert relation.temporal.grain == "year"
    
    def test_relation_with_literal_object(self):
        """Test relation with literal object value."""
        relation = Relation(
            id="R1",
            subj="E1",
            pred="has_value",
            obj=100,  # Literal value
            confidence=0.9,
            hypothesis=False
        )
        
        assert relation.obj == 100


class TestTrigger:
    """Test Trigger model."""
    
    def test_trigger_creation(self):
        """Test creating triggers."""
        trigger = Trigger(
            type=TriggerType.EVENT,
            detail="When user completes onboarding"
        )
        
        assert trigger.type == TriggerType.EVENT
        assert trigger.detail == "When user completes onboarding"


class TestCrossLink:
    """Test CrossLink model."""
    
    def test_crosslink_creation(self):
        """Test creating cross-links."""
        crosslink = CrossLink(
            from_id="QA1",
            to_id="QB2",
            type="junction"
        )
        
        assert crosslink.from_id == "QA1"
        assert crosslink.to_id == "QB2"
        assert crosslink.type == "junction"
    
    def test_crosslink_alias_field_names(self):
        """Test cross-link field name aliases."""
        # Test using 'from' and 'to' field names
        crosslink = CrossLink(**{
            "from": "QA1",
            "to": "QB2",
            "type": "junction"
        })
        
        assert crosslink.from_id == "QA1"
        assert crosslink.to_id == "QB2"


class TestMapping:
    """Test Mapping model."""
    
    def test_mapping_creation(self):
        """Test creating question-ontology mappings."""
        mapping = Mapping(
            question_id="Q1",
            mentions=["E1", "E2"],
            claims=["R1", "R2"]
        )
        
        assert mapping.question_id == "Q1"
        assert mapping.mentions == ["E1", "E2"]
        assert mapping.claims == ["R1", "R2"]
    
    def test_mapping_empty_lists(self):
        """Test mapping with empty mention/claim lists."""
        mapping = Mapping(
            question_id="Q1",
            mentions=[],
            claims=[]
        )
        
        assert mapping.mentions == []
        assert mapping.claims == []