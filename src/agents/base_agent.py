"""
Base AI Agent class providing common functionality for all agents.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import structlog
from pydantic import BaseModel, Field

from ..utils.aws_client import AWSClientManager
from ..utils.metrics import MetricsCollector


class AgentConfig(BaseModel):
    """Configuration for AI agents."""
    
    name: str
    enabled: bool = True
    execution_interval: int = Field(default=60, description="Execution interval in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    timeout: int = Field(default=300, description="Timeout in seconds")
    metrics_enabled: bool = True
    log_level: str = "INFO"


class AgentMetadata(BaseModel):
    """Metadata for agent execution."""
    
    agent_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all AI agents providing common functionality.
    """
    
    def __init__(self, config: AgentConfig, aws_client_manager: AWSClientManager):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration
            aws_client_manager: AWS client manager instance
        """
        self.config = config
        self.aws_clients = aws_client_manager
        self.metrics = MetricsCollector(enabled=config.metrics_enabled)
        
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(self.config.name)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        self._running = False
        self._execution_count = 0
    
    @abstractmethod
    async def analyze(self) -> Dict[str, Any]:
        """
        Analyze current state and determine if action is needed.
        
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    async def decide(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decisions based on analysis results.
        
        Args:
            analysis: Results from the analyze phase
            
        Returns:
            Dictionary containing decisions to execute
        """
        pass
    
    @abstractmethod
    async def execute(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the decided actions.
        
        Args:
            decisions: Decisions to execute
            
        Returns:
            Dictionary containing execution results
        """
        pass
    
    async def run_cycle(self) -> AgentMetadata:
        """
        Run a single execution cycle of the agent.
        
        Returns:
            Metadata about the execution
        """
        execution_id = f"{self.config.name}-{datetime.now(timezone.utc).isoformat()}"
        metadata = AgentMetadata(
            agent_name=self.config.name,
            execution_id=execution_id,
            start_time=datetime.now(timezone.utc)
        )
        
        try:
            self.logger.info("Starting agent execution cycle", execution_id=execution_id)
            
            with self.metrics.timer("agent_execution_time"):
                # Analysis phase
                self.logger.info("Starting analysis phase")
                analysis_results = await self._run_with_timeout(
                    self.analyze(), 
                    self.config.timeout
                )
                metadata.metrics["analysis"] = analysis_results
                
                # Decision phase
                self.logger.info("Starting decision phase")
                decisions = await self._run_with_timeout(
                    self.decide(analysis_results),
                    self.config.timeout
                )
                metadata.metrics["decisions"] = decisions
                
                # Execution phase
                if decisions.get("actions"):
                    self.logger.info("Starting execution phase")
                    execution_results = await self._run_with_timeout(
                        self.execute(decisions),
                        self.config.timeout
                    )
                    metadata.metrics["execution"] = execution_results
                else:
                    self.logger.info("No actions to execute")
                    metadata.metrics["execution"] = {"status": "no_actions"}
            
            metadata.status = "completed"
            metadata.end_time = datetime.now(timezone.utc)
            self._execution_count += 1
            
            self.logger.info(
                "Agent execution cycle completed successfully",
                execution_id=execution_id,
                duration=(metadata.end_time - metadata.start_time).total_seconds()
            )
            
        except Exception as e:
            metadata.status = "failed"
            metadata.error_message = str(e)
            metadata.end_time = datetime.now(timezone.utc)
            
            self.logger.error(
                "Agent execution cycle failed",
                execution_id=execution_id,
                error=str(e),
                exc_info=True
            )
            
            self.metrics.increment("agent_execution_errors")
            
        finally:
            self.metrics.increment("agent_execution_total")
            
        return metadata
    
    async def start(self) -> None:
        """Start the agent's continuous execution loop."""
        if self._running:
            self.logger.warning("Agent is already running")
            return
        
        self._running = True
        self.logger.info("Starting agent", interval=self.config.execution_interval)
        
        try:
            while self._running:
                if self.config.enabled:
                    await self.run_cycle()
                else:
                    self.logger.debug("Agent is disabled, skipping execution")
                
                await asyncio.sleep(self.config.execution_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Agent execution cancelled")
        except Exception as e:
            self.logger.error("Unexpected error in agent loop", error=str(e), exc_info=True)
        finally:
            self._running = False
            self.logger.info("Agent stopped")
    
    async def stop(self) -> None:
        """Stop the agent's execution loop."""
        self.logger.info("Stopping agent")
        self._running = False
    
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._running
    
    def get_execution_count(self) -> int:
        """Get the number of completed execution cycles."""
        return self._execution_count
    
    async def _run_with_timeout(self, coro, timeout: int):
        """Run a coroutine with a timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.error("Operation timed out", timeout=timeout)
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent.
        
        Returns:
            Dictionary containing health status
        """
        return {
            "agent_name": self.config.name,
            "status": "healthy" if self._running else "stopped",
            "execution_count": self._execution_count,
            "config": self.config.dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
