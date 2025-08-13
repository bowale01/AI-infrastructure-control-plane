"""
Infrastructure Management Orchestrator
Custom coordination system for autonomous AWS resource agents
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type
from enum import Enum
import uuid

from pydantic import BaseModel, Field

from ..agents.base_agent import InfrastructureAgent, InfraAgentConfig, AgentExecutionRecord
from ..agents.scaling_agent import AdaptiveScalingAgent
from ..utils.aws_client import AWSClientManager
from ..utils.metrics import MetricsCollector
from ..utils.logging_config import ContextualLogger


class CoordinationStrategy(str, Enum):
    """Available agent coordination strategies."""
    SEQUENTIAL_EXECUTION = "sequential_execution"
    CONCURRENT_EXECUTION = "concurrent_execution" 
    PRIORITY_DRIVEN = "priority_driven"


class RegisteredAgent(BaseModel):
    """Information about a registered infrastructure agent."""
    
    agent_class_name: str
    agent_configuration: InfraAgentConfig
    execution_priority: int = Field(default=1, description="Agent execution priority (higher values run first)")
    enabled: bool = True
    dependencies: List[str] = Field(default_factory=list, description="Agent dependencies")


class OrchestrationConfig(BaseModel):
    """Configuration for the control plane orchestrator."""
    
    mode: OrchestrationMode = OrchestrationMode.PARALLEL
    max_concurrent_agents: int = Field(default=5, ge=1)
    execution_timeout: int = Field(default=1800, description="Max execution time in seconds")
    health_check_interval: int = Field(default=300, description="Health check interval in seconds")
    retry_failed_agents: bool = True
    max_retries: int = Field(default=3, ge=0)


class ControlPlaneOrchestrator:
    """
    Main orchestrator for the AI agents control plane.
    
    Manages agent lifecycle, execution coordination, and state management.
    """
    
    def __init__(
        self,
        config: OrchestrationConfig,
        aws_client_manager: AWSClientManager,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the control plane orchestrator.
        
        Args:
            config: Orchestration configuration
            aws_client_manager: AWS client manager instance
            metrics_collector: Optional metrics collector
        """
        self.config = config
        self.aws_clients = aws_client_manager
        self.metrics = metrics_collector or MetricsCollector()
        self.logger = ContextualLogger("control_plane_orchestrator")
        
        # Agent management
        self.agent_registry: Dict[str, AgentRegistration] = {}
        self.active_agents: Dict[str, BaseAgent] = {}
        self.agent_metadata: Dict[str, List[AgentMetadata]] = {}
        
        # Orchestration state
        self._running = False
        self._orchestration_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._execution_count = 0
        
        # Available agent classes
        self.agent_classes = {
            'ScalingAgent': ScalingAgent,
            # Add other agent classes here as they are implemented
        }
    
    def register_agent(
        self,
        agent_name: str,
        agent_class: str,
        config: AgentConfig,
        priority: int = 1,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_name: Unique name for the agent
            agent_class: Name of the agent class
            config: Agent configuration
            priority: Agent priority (higher = more important)
            dependencies: List of agent names this agent depends on
        """
        if agent_class not in self.agent_classes:
            raise ValueError(f"Unknown agent class: {agent_class}")
        
        registration = AgentRegistration(
            agent_class=agent_class,
            config=config,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.agent_registry[agent_name] = registration
        self.agent_metadata[agent_name] = []
        
        self.logger.info(
            "Agent registered",
            agent_name=agent_name,
            agent_class=agent_class,
            priority=priority
        )
    
    def unregister_agent(self, agent_name: str) -> bool:
        """
        Unregister an agent from the orchestrator.
        
        Args:
            agent_name: Name of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_name not in self.agent_registry:
            return False
        
        # Stop the agent if it's running
        if agent_name in self.active_agents:
            asyncio.create_task(self.stop_agent(agent_name))
        
        del self.agent_registry[agent_name]
        self.agent_metadata.pop(agent_name, None)
        
        self.logger.info("Agent unregistered", agent_name=agent_name)
        return True
    
    async def start_agent(self, agent_name: str) -> bool:
        """
        Start a specific agent.
        
        Args:
            agent_name: Name of the agent to start
            
        Returns:
            True if agent was started successfully
        """
        if agent_name not in self.agent_registry:
            self.logger.error("Cannot start unregistered agent", agent_name=agent_name)
            return False
        
        if agent_name in self.active_agents:
            self.logger.warning("Agent is already running", agent_name=agent_name)
            return True
        
        try:
            registration = self.agent_registry[agent_name]
            agent_class = self.agent_classes[registration.agent_class]
            
            # Create agent instance
            agent = agent_class(registration.config, self.aws_clients)
            
            # Start the agent
            agent_task = asyncio.create_task(agent.start())
            self.active_agents[agent_name] = agent
            
            self.metrics.increment("agent_started", labels={"agent_name": agent_name})
            self.logger.info("Agent started", agent_name=agent_name)
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to start agent",
                agent_name=agent_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def stop_agent(self, agent_name: str) -> bool:
        """
        Stop a specific agent.
        
        Args:
            agent_name: Name of the agent to stop
            
        Returns:
            True if agent was stopped successfully
        """
        if agent_name not in self.active_agents:
            self.logger.warning("Agent is not running", agent_name=agent_name)
            return True
        
        try:
            agent = self.active_agents[agent_name]
            await agent.stop()
            
            del self.active_agents[agent_name]
            
            self.metrics.increment("agent_stopped", labels={"agent_name": agent_name})
            self.logger.info("Agent stopped", agent_name=agent_name)
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to stop agent",
                agent_name=agent_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def execute_orchestration_cycle(self) -> Dict[str, Any]:
        """
        Execute a single orchestration cycle.
        
        Returns:
            Results of the orchestration cycle
        """
        cycle_start = datetime.now(timezone.utc)
        cycle_id = f"orchestration-{cycle_start.isoformat()}"
        
        self.logger.info("Starting orchestration cycle", cycle_id=cycle_id)
        
        results = {
            "cycle_id": cycle_id,
            "start_time": cycle_start.isoformat(),
            "mode": self.config.mode,
            "agent_results": {},
            "errors": [],
            "summary": {}
        }
        
        try:
            # Get agents to execute based on orchestration mode
            agents_to_execute = await self._get_agents_for_execution()
            
            if self.config.mode == OrchestrationMode.SEQUENTIAL:
                await self._execute_sequential(agents_to_execute, results)
            elif self.config.mode == OrchestrationMode.PARALLEL:
                await self._execute_parallel(agents_to_execute, results)
            elif self.config.mode == OrchestrationMode.PRIORITY_BASED:
                await self._execute_priority_based(agents_to_execute, results)
            
            # Update metrics
            successful_agents = sum(
                1 for result in results["agent_results"].values()
                if result.get("status") == "completed"
            )
            
            self.metrics.increment(
                "orchestration_cycles_total",
                labels={"status": "success"}
            )
            self.metrics.set_gauge("successful_agents_last_cycle", successful_agents)
            
        except Exception as e:
            self.logger.error(
                "Orchestration cycle failed",
                cycle_id=cycle_id,
                error=str(e),
                exc_info=True
            )
            results["errors"].append(str(e))
            self.metrics.increment(
                "orchestration_cycles_total",
                labels={"status": "error"}
            )
        
        finally:
            cycle_end = datetime.now(timezone.utc)
            duration = (cycle_end - cycle_start).total_seconds()
            
            results["end_time"] = cycle_end.isoformat()
            results["duration_seconds"] = duration
            results["summary"] = {
                "total_agents": len(results["agent_results"]),
                "successful_agents": sum(
                    1 for result in results["agent_results"].values()
                    if result.get("status") == "completed"
                ),
                "failed_agents": sum(
                    1 for result in results["agent_results"].values()
                    if result.get("status") == "failed"
                ),
                "errors_count": len(results["errors"])
            }
            
            self.metrics.observe("orchestration_cycle_duration", duration)
            self._execution_count += 1
            
            self.logger.info(
                "Orchestration cycle completed",
                cycle_id=cycle_id,
                duration=duration,
                summary=results["summary"]
            )
        
        return results
    
    async def _get_agents_for_execution(self) -> List[tuple[str, BaseAgent]]:
        """Get list of agents ready for execution."""
        ready_agents = []
        
        for agent_name, registration in self.agent_registry.items():
            if not registration.enabled:
                continue
            
            if agent_name not in self.active_agents:
                # Try to start the agent if it's not running
                await self.start_agent(agent_name)
            
            if agent_name in self.active_agents:
                agent = self.active_agents[agent_name]
                
                # Check dependencies
                deps_satisfied = await self._check_dependencies(agent_name)
                
                if deps_satisfied:
                    ready_agents.append((agent_name, agent))
        
        return ready_agents
    
    async def _check_dependencies(self, agent_name: str) -> bool:
        """Check if agent dependencies are satisfied."""
        registration = self.agent_registry[agent_name]
        
        for dep_name in registration.dependencies:
            if dep_name not in self.active_agents:
                return False
            
            # Check if dependency agent has completed at least one cycle recently
            dep_metadata = self.agent_metadata.get(dep_name, [])
            if not dep_metadata:
                return False
            
            latest_metadata = max(dep_metadata, key=lambda x: x.start_time)
            if latest_metadata.status != "completed":
                return False
        
        return True
    
    async def _execute_sequential(
        self,
        agents: List[tuple[str, BaseAgent]],
        results: Dict[str, Any]
    ) -> None:
        """Execute agents sequentially."""
        # Sort by priority (highest first)
        sorted_agents = sorted(
            agents,
            key=lambda x: self.agent_registry[x[0]].priority,
            reverse=True
        )
        
        for agent_name, agent in sorted_agents:
            try:
                metadata = await asyncio.wait_for(
                    agent.run_cycle(),
                    timeout=self.config.execution_timeout
                )
                
                self.agent_metadata[agent_name].append(metadata)
                results["agent_results"][agent_name] = metadata.dict()
                
            except asyncio.TimeoutError:
                error_msg = f"Agent {agent_name} execution timed out"
                results["errors"].append(error_msg)
                results["agent_results"][agent_name] = {
                    "status": "timeout",
                    "error": error_msg
                }
            except Exception as e:
                error_msg = f"Agent {agent_name} execution failed: {str(e)}"
                results["errors"].append(error_msg)
                results["agent_results"][agent_name] = {
                    "status": "failed",
                    "error": error_msg
                }
    
    async def _execute_parallel(
        self,
        agents: List[tuple[str, BaseAgent]],
        results: Dict[str, Any]
    ) -> None:
        """Execute agents in parallel."""
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
        
        async def run_agent(agent_name: str, agent: BaseAgent):
            async with semaphore:
                try:
                    metadata = await asyncio.wait_for(
                        agent.run_cycle(),
                        timeout=self.config.execution_timeout
                    )
                    
                    self.agent_metadata[agent_name].append(metadata)
                    results["agent_results"][agent_name] = metadata.dict()
                    
                except asyncio.TimeoutError:
                    error_msg = f"Agent {agent_name} execution timed out"
                    results["errors"].append(error_msg)
                    results["agent_results"][agent_name] = {
                        "status": "timeout",
                        "error": error_msg
                    }
                except Exception as e:
                    error_msg = f"Agent {agent_name} execution failed: {str(e)}"
                    results["errors"].append(error_msg)
                    results["agent_results"][agent_name] = {
                        "status": "failed",
                        "error": error_msg
                    }
        
        # Create tasks for all agents
        tasks = [
            asyncio.create_task(run_agent(agent_name, agent))
            for agent_name, agent in agents
        ]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_priority_based(
        self,
        agents: List[tuple[str, BaseAgent]],
        results: Dict[str, Any]
    ) -> None:
        """Execute agents based on priority with dependency resolution."""
        # Group agents by priority
        priority_groups = {}
        for agent_name, agent in agents:
            priority = self.agent_registry[agent_name].priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append((agent_name, agent))
        
        # Execute in priority order (highest first)
        for priority in sorted(priority_groups.keys(), reverse=True):
            # Execute agents in this priority group in parallel
            await self._execute_parallel(priority_groups[priority], results)
    
    async def start_orchestration(self) -> None:
        """Start the orchestration loop."""
        if self._running:
            self.logger.warning("Orchestration is already running")
            return
        
        self._running = True
        self.logger.info("Starting orchestration")
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Update metrics
        self.metrics.set_gauge("active_agents", len(self.active_agents))
        
        self.logger.info("Orchestration started")
    
    async def stop_orchestration(self) -> None:
        """Stop the orchestration loop."""
        if not self._running:
            self.logger.warning("Orchestration is not running")
            return
        
        self.logger.info("Stopping orchestration")
        self._running = False
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Stop all active agents
        stop_tasks = [
            self.stop_agent(agent_name)
            for agent_name in list(self.active_agents.keys())
        ]
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.logger.info("Orchestration stopped")
    
    async def _health_check_loop(self) -> None:
        """Periodic health check of all agents."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in health check loop",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all active agents."""
        unhealthy_agents = []
        
        for agent_name, agent in self.active_agents.items():
            try:
                health_status = await agent.health_check()
                
                if health_status.get("status") != "healthy":
                    unhealthy_agents.append(agent_name)
                    self.logger.warning(
                        "Agent health check failed",
                        agent_name=agent_name,
                        health_status=health_status
                    )
                else:
                    self.metrics.set_gauge(
                        "agent_health",
                        1,
                        labels={"agent_name": agent_name}
                    )
                    
            except Exception as e:
                unhealthy_agents.append(agent_name)
                self.logger.error(
                    "Agent health check error",
                    agent_name=agent_name,
                    error=str(e)
                )
                self.metrics.set_gauge(
                    "agent_health",
                    0,
                    labels={"agent_name": agent_name}
                )
        
        # Restart unhealthy agents if configured
        if self.config.retry_failed_agents:
            for agent_name in unhealthy_agents:
                await self._restart_agent(agent_name)
    
    async def _restart_agent(self, agent_name: str) -> None:
        """Restart an unhealthy agent."""
        self.logger.info("Restarting unhealthy agent", agent_name=agent_name)
        
        try:
            await self.stop_agent(agent_name)
            await asyncio.sleep(5)  # Wait before restart
            await self.start_agent(agent_name)
            
            self.metrics.increment(
                "agent_restarts_total",
                labels={"agent_name": agent_name}
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to restart agent",
                agent_name=agent_name,
                error=str(e),
                exc_info=True
            )
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """
        Get current orchestration status.
        
        Returns:
            Status information
        """
        return {
            "running": self._running,
            "execution_count": self._execution_count,
            "registered_agents": len(self.agent_registry),
            "active_agents": len(self.active_agents),
            "config": self.config.dict(),
            "agents": {
                name: {
                    "status": "active" if name in self.active_agents else "inactive",
                    "priority": reg.priority,
                    "enabled": reg.enabled,
                    "executions": len(self.agent_metadata.get(name, []))
                }
                for name, reg in self.agent_registry.items()
            }
        }
