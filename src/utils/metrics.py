"""
Metrics collection and monitoring utilities.
"""
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import threading


class MetricsCollector:
    """
    Centralized metrics collection for monitoring agent performance.
    """
    
    def __init__(self, enabled: bool = True, registry: Optional[CollectorRegistry] = None):
        """
        Initialize the metrics collector.
        
        Args:
            enabled: Whether metrics collection is enabled
            registry: Prometheus registry to use (creates new if None)
        """
        self.enabled = enabled
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        if self.enabled:
            self._setup_default_metrics()
    
    def _setup_default_metrics(self) -> None:
        """Setup default metrics for agent monitoring."""
        # Counters
        self._metrics['agent_execution_total'] = Counter(
            'agent_execution_total',
            'Total number of agent executions',
            ['agent_name', 'status'],
            registry=self.registry
        )
        
        self._metrics['agent_execution_errors'] = Counter(
            'agent_execution_errors_total',
            'Total number of agent execution errors',
            ['agent_name', 'error_type'],
            registry=self.registry
        )
        
        self._metrics['aws_api_calls_total'] = Counter(
            'aws_api_calls_total',
            'Total number of AWS API calls',
            ['service', 'operation', 'status'],
            registry=self.registry
        )
        
        # Histograms
        self._metrics['agent_execution_time'] = Histogram(
            'agent_execution_duration_seconds',
            'Time spent executing agent cycles',
            ['agent_name'],
            registry=self.registry
        )
        
        self._metrics['aws_api_call_duration'] = Histogram(
            'aws_api_call_duration_seconds',
            'Time spent on AWS API calls',
            ['service', 'operation'],
            registry=self.registry
        )
        
        # Gauges
        self._metrics['active_agents'] = Gauge(
            'active_agents',
            'Number of currently active agents',
            registry=self.registry
        )
        
        self._metrics['scaling_targets'] = Gauge(
            'scaling_targets_total',
            'Total number of scaling targets',
            ['agent_name'],
            registry=self.registry
        )
        
        self._metrics['current_capacity'] = Gauge(
            'current_capacity',
            'Current capacity of scaling targets',
            ['target_id', 'resource_type'],
            registry=self.registry
        )
    
    def increment(self, metric_name: str, labels: Optional[Dict[str, str]] = None, value: float = 1) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            labels: Labels to apply to the metric
            value: Value to increment by (default: 1)
        """
        if not self.enabled:
            return
        
        try:
            if metric_name in self._metrics:
                metric = self._metrics[metric_name]
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
        except Exception as e:
            # Don't let metrics errors break the application
            print(f"Error incrementing metric {metric_name}: {e}")
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            metric_name: Name of the metric
            value: Value to set
            labels: Labels to apply to the metric
        """
        if not self.enabled:
            return
        
        try:
            if metric_name in self._metrics:
                metric = self._metrics[metric_name]
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
        except Exception as e:
            print(f"Error setting gauge {metric_name}: {e}")
    
    def observe(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value for a histogram metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to observe
            labels: Labels to apply to the metric
        """
        if not self.enabled:
            return
        
        try:
            if metric_name in self._metrics:
                metric = self._metrics[metric_name]
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
        except Exception as e:
            print(f"Error observing metric {metric_name}: {e}")
    
    @contextmanager
    def timer(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            metric_name: Name of the metric to time
            labels: Labels to apply to the metric
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe(metric_name, duration, labels)
    
    def record_aws_api_call(
        self, 
        service: str, 
        operation: str, 
        duration: float, 
        success: bool = True
    ) -> None:
        """
        Record metrics for an AWS API call.
        
        Args:
            service: AWS service name
            operation: API operation name
            duration: Call duration in seconds
            success: Whether the call was successful
        """
        if not self.enabled:
            return
        
        status = 'success' if success else 'error'
        
        self.increment(
            'aws_api_calls_total',
            labels={'service': service, 'operation': operation, 'status': status}
        )
        
        self.observe(
            'aws_api_call_duration',
            duration,
            labels={'service': service, 'operation': operation}
        )
    
    def record_agent_execution(
        self, 
        agent_name: str, 
        duration: float, 
        success: bool = True
    ) -> None:
        """
        Record metrics for an agent execution.
        
        Args:
            agent_name: Name of the agent
            duration: Execution duration in seconds
            success: Whether the execution was successful
        """
        if not self.enabled:
            return
        
        status = 'success' if success else 'error'
        
        self.increment(
            'agent_execution_total',
            labels={'agent_name': agent_name, 'status': status}
        )
        
        self.observe(
            'agent_execution_time',
            duration,
            labels={'agent_name': agent_name}
        )
        
        if not success:
            self.increment(
                'agent_execution_errors',
                labels={'agent_name': agent_name, 'error_type': 'execution_failure'}
            )
    
    def update_scaling_metrics(
        self, 
        agent_name: str, 
        target_id: str, 
        resource_type: str, 
        current_capacity: int,
        total_targets: int
    ) -> None:
        """
        Update scaling-related metrics.
        
        Args:
            agent_name: Name of the scaling agent
            target_id: ID of the scaling target
            resource_type: Type of resource being scaled
            current_capacity: Current capacity of the target
            total_targets: Total number of scaling targets
        """
        if not self.enabled:
            return
        
        self.set_gauge(
            'current_capacity',
            current_capacity,
            labels={'target_id': target_id, 'resource_type': resource_type}
        )
        
        self.set_gauge(
            'scaling_targets',
            total_targets,
            labels={'agent_name': agent_name}
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        if not self.enabled:
            return {"enabled": False}
        
        summary = {
            "enabled": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_count": len(self._metrics),
            "registry_collectors": len(list(self.registry._collector_to_names.keys()))
        }
        
        return summary
    
    def create_custom_counter(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None
    ) -> Optional[Counter]:
        """
        Create a custom counter metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: List of label names
            
        Returns:
            Counter instance or None if disabled
        """
        if not self.enabled:
            return None
        
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(
                    name,
                    description,
                    labels or [],
                    registry=self.registry
                )
            return self._metrics[name]
    
    def create_custom_gauge(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None
    ) -> Optional[Gauge]:
        """
        Create a custom gauge metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: List of label names
            
        Returns:
            Gauge instance or None if disabled
        """
        if not self.enabled:
            return None
        
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(
                    name,
                    description,
                    labels or [],
                    registry=self.registry
                )
            return self._metrics[name]
    
    def create_custom_histogram(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None,
        buckets: Optional[tuple] = None
    ) -> Optional[Histogram]:
        """
        Create a custom histogram metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: List of label names
            buckets: Custom bucket configuration
            
        Returns:
            Histogram instance or None if disabled
        """
        if not self.enabled:
            return None
        
        with self._lock:
            if name not in self._metrics:
                kwargs = {
                    'name': name,
                    'documentation': description,
                    'labelnames': labels or [],
                    'registry': self.registry
                }
                
                if buckets:
                    kwargs['buckets'] = buckets
                
                self._metrics[name] = Histogram(**kwargs)
            return self._metrics[name]


# Global metrics collector instance
metrics_collector = MetricsCollector()
