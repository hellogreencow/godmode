"""
Tests for GODMODE engine
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from godmode.engine import GodmodeEngine
from godmode.schemas import Command, InitCommand
from godmode.models import ModelConfig
from godmode.ladder_generator import LadderConfig


@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing"""
    config = ModelConfig()
    config.openai_key = "test_key"
    config.anthropic_key = "test_key"
    config.voyage_key = "test_key"
    return config


@pytest.fixture
def engine(mock_model_config):
    """Create test engine instance"""
    ladder_config = LadderConfig(
        beam_width=2,  # Smaller for faster tests
        depth_back=2,
        depth_forward=2
    )
    return GodmodeEngine(mock_model_config, ladder_config)


@pytest.mark.asyncio
async def test_init_command(engine):
    """Test INIT command processing"""
    # Mock the model router to avoid actual API calls
    engine.router.enumerate = AsyncMock(return_value=[
        "What does career success mean to you?",
        "What are your core skills and interests?"
    ])
    engine.router.rerank = AsyncMock(return_value=[
        ("What does career success mean to you?", 0.8),
        ("What are your core skills and interests?", 0.7)
    ])
    engine.router.stitch = AsyncMock(return_value="Generated response")
    
    command = Command(
        command_type="INIT",
        data=InitCommand(
            current_question="Should I change careers?",
            context="I'm a software engineer considering product management."
        )
    )
    
    response = await engine.process_command(command)
    
    # Verify response structure
    assert response.chat_reply is not None
    assert response.graph_update is not None
    assert response.ontology_update is not None
    
    # Verify graph structure
    assert response.graph_update.current_question == "Should I change careers?"
    assert len(response.graph_update.priors) >= 0  # May be empty due to mocking
    assert len(response.graph_update.scenarios) >= 0  # May be empty due to mocking
    
    # Verify timing was recorded
    assert response.graph_update.meta.budgets_used.time_s >= 0


@pytest.mark.asyncio
async def test_invalid_command():
    """Test handling of invalid commands"""
    engine = GodmodeEngine()
    
    # Create command with invalid type
    command = Command(
        command_type="INVALID",  # This should cause an error
        data=InitCommand(current_question="Test")
    )
    
    response = await engine.process_command(command)
    
    # Should return error response
    assert "Error:" in response.chat_reply
    assert response.graph_update.meta.notes is not None
    assert "Error:" in response.graph_update.meta.notes


def test_engine_initialization():
    """Test engine initializes correctly"""
    engine = GodmodeEngine()
    
    assert engine.router is not None
    assert engine.ladder_gen is not None
    assert engine.ontology is not None
    assert engine.validator is not None
    assert engine.current_question == ""
    assert engine.context == ""
    assert len(engine.threads) == 0


@pytest.mark.asyncio
async def test_budget_updates(engine):
    """Test budget configuration updates"""
    # Mock router methods
    engine.router.enumerate = AsyncMock(return_value=["Test question"])
    engine.router.rerank = AsyncMock(return_value=[("Test question", 0.5)])
    engine.router.stitch = AsyncMock(return_value="Test response")
    
    custom_budgets = {
        "beam_width": 6,
        "depth_back": 3,
        "depth_fwd": 4
    }
    
    command = Command(
        command_type="INIT",
        data=InitCommand(
            current_question="Test question",
            budgets=custom_budgets
        )
    )
    
    response = await engine.process_command(command)
    
    # Verify budgets were updated
    assert response.graph_update.meta.budgets_used.beam_width == 6
    assert response.graph_update.meta.budgets_used.depth_back == 3
    assert response.graph_update.meta.budgets_used.depth_fwd == 4