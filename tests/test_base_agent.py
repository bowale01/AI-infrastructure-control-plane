"""
Tests for the base agent functionality.
"""
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from src.agents.base_agent import BaseAgent, AgentConfig, AgentMetadata
from src.utils.aws_client import AWSClientManager
from src.utils.metrics import MetricsCollector


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing."""
    
    def __init__(self, config, aws_client_manager):
        super().__init__(config, aws_client_manager)
        self.analyze_called = False
        self.decide_called = False
        self.execute_called = False
        
    async def analyze(self):
        self.analyze_called = True
        return {"status": "analyzed", "metrics": {"cpu": 50}}
    
    async def decide(self, analysis):
        self.decide_called = True
        return {"actions": [{"type": "scale_up", "amount": 1}]}
    
    async def execute(self, decisions):
        self.execute_called = True
        return {"status": "executed", "actions_completed": len(decisions.get("actions", []))}


@pytest.fixture
def mock_aws_client_manager():
    """Mock AWS client manager."""
    return Mock(spec=AWSClientManager)


@pytest.fixture
def agent_config():
    """Default agent configuration for testing."""
    return AgentConfig(
        name="test_agent",
        enabled=True,
        execution_interval=60,
        retry_attempts=3,
        timeout=300,
        metrics_enabled=True,
        log_level="INFO"
    )


@pytest.fixture
def test_agent(agent_config, mock_aws_client_manager):
    """Test agent instance."""
    return TestAgent(agent_config, mock_aws_client_manager)


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    def test_agent_initialization(self, test_agent, agent_config):
        """Test agent initialization."""
        assert test_agent.config == agent_config
        assert test_agent._running is False
        assert test_agent._execution_count == 0
        assert test_agent.logger is not None
        assert isinstance(test_agent.metrics, MetricsCollector)
    
    @pytest.mark.asyncio
    async def test_run_cycle_success(self, test_agent):
        """Test successful agent execution cycle."""
        metadata = await test_agent.run_cycle()
        
        assert isinstance(metadata, AgentMetadata)
        assert metadata.agent_name == "test_agent"
        assert metadata.status == "completed"
        assert metadata.end_time is not None
        assert test_agent.analyze_called
        assert test_agent.decide_called
        assert test_agent.execute_called
        assert test_agent._execution_count == 1
    
    @pytest.mark.asyncio
    async def test_run_cycle_analyze_failure(self, test_agent):
        """Test agent cycle with analysis failure."""
        # Override analyze to raise an exception
        async def failing_analyze():
            raise Exception("Analysis failed")
        
        test_agent.analyze = failing_analyze
        
        metadata = await test_agent.run_cycle()
        
        assert metadata.status == "failed"
        assert "Analysis failed" in metadata.error_message
        assert metadata.end_time is not None
    
    @pytest.mark.asyncio
    async def test_run_cycle_no_actions(self, test_agent):
        """Test agent cycle when no actions are decided."""
        # Override decide to return no actions
        async def decide_no_actions(analysis):
            test_agent.decide_called = True
            return {"actions": []}
        
        test_agent.decide = decide_no_actions
        
        metadata = await test_agent.run_cycle()
        
        assert metadata.status == "completed"
        assert test_agent.analyze_called
        assert test_agent.decide_called
        assert not test_agent.execute_called  # Should not execute if no actions
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_agent):
        """Test agent health check."""
        health_status = await test_agent.health_check()
        
        assert isinstance(health_status, dict)
        assert health_status["agent_name"] == "test_agent"
        assert health_status["status"] == "stopped"  # Agent not running
        assert health_status["execution_count"] == 0
        assert "config" in health_status
        assert "timestamp" in health_status
    
    def test_is_running_initially_false(self, test_agent):
        """Test that agent is not running initially."""
        assert test_agent.is_running() is False
    
    def test_get_execution_count_initially_zero(self, test_agent):
        """Test that execution count is initially zero."""
        assert test_agent.get_execution_count() == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_agent(self, test_agent):
        """Test starting and stopping the agent."""
        # Start the agent in a task
        start_task = asyncio.create_task(test_agent.start())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        assert test_agent.is_running() is True
        
        # Stop the agent
        await test_agent.stop()
        
        # Wait for the start task to complete
        await asyncio.sleep(0.1)
        start_task.cancel()
        
        try:
            await start_task
        except asyncio.CancelledError:
            pass
        
        assert test_agent.is_running() is False


class TestAgentConfig:
    """Test cases for AgentConfig model."""
    
    def test_agent_config_defaults(self):
        """Test agent configuration with defaults."""
        config = AgentConfig(name="test")
        
        assert config.name == "test"
        assert config.enabled is True
        assert config.execution_interval == 60
        assert config.retry_attempts == 3
        assert config.timeout == 300
        assert config.metrics_enabled is True
        assert config.log_level == "INFO"
    
    def test_agent_config_custom_values(self):
        """Test agent configuration with custom values."""
        config = AgentConfig(
            name="custom_agent",
            enabled=False,
            execution_interval=120,
            retry_attempts=5,
            timeout=600,
            metrics_enabled=False,
            log_level="DEBUG"
        )
        
        assert config.name == "custom_agent"
        assert config.enabled is False
        assert config.execution_interval == 120
        assert config.retry_attempts == 5
        assert config.timeout == 600
        assert config.metrics_enabled is False
        assert config.log_level == "DEBUG"


class TestAgentMetadata:
    """Test cases for AgentMetadata model."""
    
    def test_agent_metadata_creation(self):
        """Test agent metadata creation."""
        start_time = datetime.now(timezone.utc)
        
        metadata = AgentMetadata(
            agent_name="test_agent",
            execution_id="test-123",
            start_time=start_time
        )
        
        assert metadata.agent_name == "test_agent"
        assert metadata.execution_id == "test-123"
        assert metadata.start_time == start_time
        assert metadata.end_time is None
        assert metadata.status == "running"
        assert metadata.error_message is None
        assert metadata.metrics == {}
    
    def test_agent_metadata_completion(self):
        """Test agent metadata on completion."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        
        metadata = AgentMetadata(
            agent_name="test_agent",
            execution_id="test-123",
            start_time=start_time,
            end_time=end_time,
            status="completed",
            metrics={"analysis": {"cpu": 50}, "execution": {"actions": 1}}
        )
        
        assert metadata.status == "completed"
        assert metadata.end_time == end_time
        assert "analysis" in metadata.metrics
        assert "execution" in metadata.metrics
