"""
Main application entry point for Infrastructure Automation Agents.
"""
import asyncio
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from src.agents.scaling_agent import AdaptiveScalingAgent, ManagedResource, ResourceThreshold
from src.control_plane.orchestrator import (
    InfrastructureOrchestrator, 
    OrchestrationConfiguration, 
    CoordinationStrategy
)
from src.agents.base_agent import InfraAgentConfig
from src.utils.aws_client import AWSClientManager
from src.utils.metrics import MetricsCollector
from src.utils.logging_config import setup_logging, get_logger


class InfraAutomationApplication:
    """
    Main application class for Infrastructure Automation Agents.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Agents application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/agents_config.yaml"
        self.config: Dict[str, Any] = {}
        self.orchestrator: Optional[ControlPlaneOrchestrator] = None
        self.aws_client_manager: Optional[AWSClientManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.logger = None
        self._shutdown_event = asyncio.Event()
        
    def load_configuration(self) -> None:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            print(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        logging_config = self.config.get('logging', {})
        
        setup_logging(
            log_level=logging_config.get('level', 'INFO'),
            enable_json=logging_config.get('format') == 'json',
            enable_console=logging_config.get('console_enabled', True),
            log_file=logging_config.get('file_path') if logging_config.get('file_enabled') else None
        )
        
        self.logger = get_logger("ai_agents_app")
        self.logger.info("Logging configured")
    
    def initialize_aws_clients(self) -> None:
        """Initialize AWS client manager."""
        aws_config = self.config.get('aws', {})
        
        self.aws_client_manager = AWSClientManager(
            region_name=aws_config.get('region', 'us-east-1')
        )
        
        # Test AWS connectivity
        health_check = self.aws_client_manager.health_check()
        if health_check['status'] == 'healthy':
            self.logger.info(
                "AWS connectivity established",
                account_id=health_check.get('account_id'),
                region=health_check.get('region')
            )
        else:
            self.logger.error("AWS connectivity failed", error=health_check.get('error'))
            sys.exit(1)
    
    def initialize_metrics(self) -> None:
        """Initialize metrics collection."""
        metrics_config = self.config.get('metrics', {})
        
        self.metrics_collector = MetricsCollector(
            enabled=metrics_config.get('enabled', True)
        )
        
        self.logger.info("Metrics collection initialized")
    
    def setup_orchestrator(self) -> None:
        """Setup the control plane orchestrator."""
        control_plane_config = self.config.get('control_plane', {})
        orchestration_config = control_plane_config.get('orchestration', {})
        
        try:
            config = OrchestrationConfig(
                mode=OrchestrationMode(orchestration_config.get('mode', 'parallel')),
                max_concurrent_agents=orchestration_config.get('max_concurrent_agents', 5),
                execution_timeout=orchestration_config.get('execution_timeout', 1800),
                health_check_interval=orchestration_config.get('health_check_interval', 300),
                retry_failed_agents=orchestration_config.get('retry_failed_agents', True),
                max_retries=orchestration_config.get('max_retries', 3)
            )
            
            self.orchestrator = ControlPlaneOrchestrator(
                config=config,
                aws_client_manager=self.aws_client_manager,
                metrics_collector=self.metrics_collector
            )
            
            self.logger.info("Control plane orchestrator initialized")
            
        except ValidationError as e:
            self.logger.error("Invalid orchestration configuration", error=str(e))
            sys.exit(1)
    
    def register_agents(self) -> None:
        """Register agents with the orchestrator."""
        agents_config = self.config.get('agents', {})
        priorities = self.config.get('control_plane', {}).get('agent_priorities', {})
        dependencies = self.config.get('control_plane', {}).get('dependencies', {})
        
        for agent_name, agent_config in agents_config.items():
            try:
                # Create agent configuration
                config = AgentConfig(**agent_config)
                
                # Determine agent class based on name
                agent_class = self._get_agent_class(agent_name)
                
                # Register with orchestrator
                self.orchestrator.register_agent(
                    agent_name=agent_name,
                    agent_class=agent_class,
                    config=config,
                    priority=priorities.get(agent_name, 1),
                    dependencies=dependencies.get(agent_name, [])
                )
                
                # Configure agent-specific settings
                self._configure_agent(agent_name, agent_config)
                
                self.logger.info(
                    "Agent registered",
                    agent_name=agent_name,
                    agent_class=agent_class,
                    enabled=config.enabled
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to register agent",
                    agent_name=agent_name,
                    error=str(e)
                )
    
    def _get_agent_class(self, agent_name: str) -> str:
        """Get agent class name based on agent name."""
        agent_class_mapping = {
            'scaling_agent': 'ScalingAgent',
            'recovery_agent': 'RecoveryAgent',
            'diagnostics_agent': 'DiagnosticsAgent'
        }
        
        return agent_class_mapping.get(agent_name, 'ScalingAgent')
    
    def _configure_agent(self, agent_name: str, agent_config: Dict[str, Any]) -> None:
        """Configure agent-specific settings."""
        if agent_name == 'scaling_agent' and 'scaling_targets' in agent_config:
            self._configure_scaling_agent(agent_config['scaling_targets'])
    
    def _configure_scaling_agent(self, targets_config: list) -> None:
        """Configure scaling agent targets."""
        # This would be called after the agent is created and started
        # For now, we'll store the configuration for later use
        self.scaling_targets_config = targets_config
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self) -> None:
        """Start the AI Agents application."""
        try:
            self.logger.info("Starting AI Agents application")
            
            # Start the orchestrator
            await self.orchestrator.start_orchestration()
            
            # Configure scaling agent after it's started
            if hasattr(self, 'scaling_targets_config'):
                await self._apply_scaling_targets_config()
            
            self.logger.info("AI Agents application started successfully")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except Exception as e:
            self.logger.error("Error starting application", error=str(e), exc_info=True)
            raise
    
    async def _apply_scaling_targets_config(self) -> None:
        """Apply scaling targets configuration to the scaling agent."""
        try:
            # Get the scaling agent instance
            scaling_agent = None
            if self.orchestrator and 'scaling_agent' in self.orchestrator.active_agents:
                scaling_agent = self.orchestrator.active_agents['scaling_agent']
            
            if scaling_agent and hasattr(self, 'scaling_targets_config'):
                for target_config in self.scaling_targets_config:
                    # Create scaling rules
                    scaling_rules = []
                    for rule_config in target_config.get('scaling_rules', []):
                        rule = ScalingRule(**rule_config)
                        scaling_rules.append(rule)
                    
                    # Create scaling target
                    target = ScalingTarget(
                        resource_type=target_config['resource_type'],
                        resource_id=target_config['resource_id'],
                        min_capacity=target_config['min_capacity'],
                        max_capacity=target_config['max_capacity'],
                        scaling_rules=scaling_rules
                    )
                    
                    # Add target to scaling agent
                    scaling_agent.add_scaling_target(target)
                
                self.logger.info(
                    "Scaling targets configured",
                    targets_count=len(self.scaling_targets_config)
                )
                
        except Exception as e:
            self.logger.error("Error configuring scaling targets", error=str(e))
    
    async def shutdown(self) -> None:
        """Shutdown the AI Agents application."""
        try:
            self.logger.info("Shutting down AI Agents application")
            
            if self.orchestrator:
                await self.orchestrator.stop_orchestration()
            
            self.logger.info("AI Agents application shutdown complete")
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e), exc_info=True)
        finally:
            self._shutdown_event.set()
    
    async def run(self) -> None:
        """Run the complete AI Agents application lifecycle."""
        try:
            # Load configuration
            self.load_configuration()
            
            # Setup logging
            self.setup_logging()
            
            # Initialize components
            self.initialize_aws_clients()
            self.initialize_metrics()
            self.setup_orchestrator()
            
            # Register agents
            self.register_agents()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Start the application
            await self.start()
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            if self.logger:
                self.logger.error("Application failed", error=str(e), exc_info=True)
            else:
                print(f"Application failed: {e}")
            sys.exit(1)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agents for AWS Infrastructure Automation")
    parser.add_argument(
        "--config",
        type=str,
        default="config/agents_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    app = AIAgentsApplication(config_path=args.config)
    await app.run()


def lambda_handler(event, context):
    """
    AWS Lambda handler for the AI Agents system.
    
    Args:
        event: Lambda event data
        context: Lambda context object
        
    Returns:
        Dict: Response with status and results
    """
    import json
    import logging
    
    # Setup basic logging for Lambda
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"AI Agents Lambda triggered with event: {json.dumps(event)}")
        
        # For now, return a simple response
        # In a full implementation, you would:
        # 1. Parse the event to determine what action to take
        # 2. Initialize the appropriate agent(s)
        # 3. Execute the scaling/monitoring logic
        # 4. Return results
        
        response = {
            'statusCode': 200,
            'body': {
                'message': 'AI Agents system executed successfully',
                'function_name': context.function_name,
                'request_id': context.aws_request_id,
                'event_processed': event,
                'environment': 'development'
            }
        }
        
        logger.info(f"AI Agents execution completed: {response}")
        return response
        
    except Exception as e:
        logger.error(f"AI Agents execution failed: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'message': 'AI Agents execution failed',
                'request_id': context.aws_request_id
            }
        }


if __name__ == "__main__":
    asyncio.run(main())
