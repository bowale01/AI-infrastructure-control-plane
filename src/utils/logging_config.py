"""
Logging configuration for structured logging across the application.
"""
import logging
import sys
from typing import Any, Dict, Optional
import structlog
from datetime import datetime, timezone


def setup_logging(
    log_level: str = "INFO",
    enable_json: bool = True,
    enable_console: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting
        enable_console: Whether to enable console logging
        log_file: Optional log file path
    """
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        handlers=[]
    )
    
    # Processors for structured logging
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Bound logger instance
    """
    return structlog.get_logger(name)


class AWSLoggingHandler:
    """
    Handler for sending logs to AWS CloudWatch Logs.
    """
    
    def __init__(self, log_group: str, log_stream: str, aws_client_manager):
        """
        Initialize the AWS logging handler.
        
        Args:
            log_group: CloudWatch log group name
            log_stream: CloudWatch log stream name
            aws_client_manager: AWS client manager instance
        """
        self.log_group = log_group
        self.log_stream = log_stream
        self.cloudwatch_logs = aws_client_manager.get_client('logs')
        self.sequence_token = None
        
        # Ensure log group and stream exist
        self._ensure_log_group_exists()
        self._ensure_log_stream_exists()
    
    def _ensure_log_group_exists(self) -> None:
        """Ensure the CloudWatch log group exists."""
        try:
            self.cloudwatch_logs.create_log_group(logGroupName=self.log_group)
        except self.cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
            pass  # Log group already exists
    
    def _ensure_log_stream_exists(self) -> None:
        """Ensure the CloudWatch log stream exists."""
        try:
            self.cloudwatch_logs.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=self.log_stream
            )
        except self.cloudwatch_logs.exceptions.ResourceAlreadyExistsException:
            # Get the sequence token for existing stream
            response = self.cloudwatch_logs.describe_log_streams(
                logGroupName=self.log_group,
                logStreamNamePrefix=self.log_stream
            )
            
            if response['logStreams']:
                self.sequence_token = response['logStreams'][0].get('uploadSequenceToken')
    
    def send_log(self, message: str, level: str = "INFO") -> None:
        """
        Send a log message to CloudWatch.
        
        Args:
            message: Log message
            level: Log level
        """
        try:
            log_event = {
                'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
                'message': f"[{level}] {message}"
            }
            
            kwargs = {
                'logGroupName': self.log_group,
                'logStreamName': self.log_stream,
                'logEvents': [log_event]
            }
            
            if self.sequence_token:
                kwargs['sequenceToken'] = self.sequence_token
            
            response = self.cloudwatch_logs.put_log_events(**kwargs)
            self.sequence_token = response['nextSequenceToken']
            
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Error sending log to CloudWatch: {e}")


class ContextualLogger:
    """
    Logger wrapper that maintains context across operations.
    """
    
    def __init__(self, name: str, initial_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the contextual logger.
        
        Args:
            name: Logger name
            initial_context: Initial context to bind
        """
        self.logger = get_logger(name)
        self.context = initial_context or {}
    
    def bind(self, **kwargs) -> 'ContextualLogger':
        """
        Bind additional context to the logger.
        
        Args:
            **kwargs: Context key-value pairs
            
        Returns:
            New logger instance with bound context
        """
        new_context = {**self.context, **kwargs}
        new_logger = ContextualLogger(self.logger.name, new_context)
        new_logger.logger = self.logger.bind(**new_context)
        return new_logger
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self.logger.critical(message, **kwargs)


def log_aws_api_call(
    logger: structlog.stdlib.BoundLogger,
    service: str,
    operation: str,
    duration: float,
    success: bool = True,
    **kwargs
) -> None:
    """
    Log an AWS API call with standardized format.
    
    Args:
        logger: Logger instance
        service: AWS service name
        operation: API operation name
        duration: Call duration in seconds
        success: Whether the call was successful
        **kwargs: Additional context
    """
    log_data = {
        'aws_service': service,
        'aws_operation': operation,
        'duration_seconds': duration,
        'success': success,
        **kwargs
    }
    
    if success:
        logger.info("AWS API call completed", **log_data)
    else:
        logger.error("AWS API call failed", **log_data)


def log_agent_execution(
    logger: structlog.stdlib.BoundLogger,
    agent_name: str,
    execution_id: str,
    phase: str,
    duration: Optional[float] = None,
    success: bool = True,
    **kwargs
) -> None:
    """
    Log an agent execution phase with standardized format.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        execution_id: Unique execution identifier
        phase: Execution phase (analyze, decide, execute)
        duration: Phase duration in seconds
        success: Whether the phase was successful
        **kwargs: Additional context
    """
    log_data = {
        'agent_name': agent_name,
        'execution_id': execution_id,
        'phase': phase,
        'success': success,
        **kwargs
    }
    
    if duration is not None:
        log_data['duration_seconds'] = duration
    
    if success:
        logger.info(f"Agent {phase} phase completed", **log_data)
    else:
        logger.error(f"Agent {phase} phase failed", **log_data)


# Default logger instance
default_logger = get_logger("ai_agents")
