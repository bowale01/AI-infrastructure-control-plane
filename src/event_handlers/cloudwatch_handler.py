"""
CloudWatch Event Handler for processing CloudWatch alarms and metrics.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.aws_client import AWSClientManager
from ..utils.logging_config import ContextualLogger
from ..utils.metrics import MetricsCollector


class CloudWatchAlarm(BaseModel):
    """CloudWatch alarm data model."""
    
    alarm_name: str
    alarm_description: Optional[str] = None
    state_value: str
    state_reason: str
    state_timestamp: datetime
    metric_name: str
    namespace: str
    dimensions: Dict[str, str] = Field(default_factory=dict)
    threshold: float
    comparison_operator: str
    evaluation_periods: int
    period: int


class CloudWatchMetric(BaseModel):
    """CloudWatch metric data model."""
    
    metric_name: str
    namespace: str
    dimensions: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime
    value: float
    unit: str


class CloudWatchEventHandler:
    """
    Handler for processing CloudWatch events, alarms, and metrics.
    """
    
    def __init__(self, aws_client_manager: AWSClientManager):
        """
        Initialize the CloudWatch event handler.
        
        Args:
            aws_client_manager: AWS client manager instance
        """
        self.aws_clients = aws_client_manager
        self.cloudwatch = self.aws_clients.get_client('cloudwatch')
        self.events = self.aws_clients.get_client('events')
        self.logger = ContextualLogger("cloudwatch_event_handler")
        self.metrics = MetricsCollector()
        
        # Event processing stats
        self.processed_events = 0
        self.alarm_events = 0
        self.metric_events = 0
    
    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a CloudWatch event.
        
        Args:
            event: Event payload from EventBridge
            
        Returns:
            Processing result
        """
        try:
            event_source = event.get('source', '')
            detail_type = event.get('detail-type', '')
            
            self.logger.info(
                "Processing CloudWatch event",
                source=event_source,
                detail_type=detail_type,
                event_id=event.get('id')
            )
            
            result = {"status": "success", "actions": []}
            
            if event_source == 'aws.cloudwatch' and 'Alarm' in detail_type:
                result = await self._process_alarm_event(event)
            elif event_source == 'aws.cloudwatch' and 'Metric' in detail_type:
                result = await self._process_metric_event(event)
            else:
                result = {"status": "ignored", "reason": "Unsupported event type"}
            
            self.processed_events += 1
            self.metrics.increment(
                "cloudwatch_events_processed",
                labels={"source": event_source, "status": result["status"]}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Error processing CloudWatch event",
                event=event,
                error=str(e),
                exc_info=True
            )
            
            self.metrics.increment(
                "cloudwatch_events_errors",
                labels={"error_type": "processing_error"}
            )
            
            return {"status": "error", "error": str(e)}
    
    async def _process_alarm_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process CloudWatch alarm state change events."""
        try:
            detail = event.get('detail', {})
            
            alarm = CloudWatchAlarm(
                alarm_name=detail.get('alarmName', ''),
                alarm_description=detail.get('alarmDescription'),
                state_value=detail.get('state', {}).get('value', ''),
                state_reason=detail.get('state', {}).get('reason', ''),
                state_timestamp=datetime.fromisoformat(
                    detail.get('state', {}).get('timestamp', '').replace('Z', '+00:00')
                ),
                metric_name=detail.get('metricName', ''),
                namespace=detail.get('namespace', ''),
                dimensions=detail.get('dimensions', {}),
                threshold=float(detail.get('threshold', 0)),
                comparison_operator=detail.get('comparisonOperator', ''),
                evaluation_periods=int(detail.get('evaluationPeriods', 1)),
                period=int(detail.get('period', 300))
            )
            
            self.alarm_events += 1
            
            # Determine actions based on alarm state
            actions = await self._determine_alarm_actions(alarm)
            
            self.logger.info(
                "Processed alarm event",
                alarm_name=alarm.alarm_name,
                state=alarm.state_value,
                actions_count=len(actions)
            )
            
            return {
                "status": "processed",
                "alarm": alarm.dict(),
                "actions": actions
            }
            
        except Exception as e:
            self.logger.error(
                "Error processing alarm event",
                detail=event.get('detail'),
                error=str(e)
            )
            raise
    
    async def _process_metric_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process CloudWatch metric events."""
        try:
            detail = event.get('detail', {})
            
            metric = CloudWatchMetric(
                metric_name=detail.get('metricName', ''),
                namespace=detail.get('namespace', ''),
                dimensions=detail.get('dimensions', {}),
                timestamp=datetime.fromisoformat(
                    detail.get('timestamp', '').replace('Z', '+00:00')
                ),
                value=float(detail.get('value', 0)),
                unit=detail.get('unit', '')
            )
            
            self.metric_events += 1
            
            # Analyze metric for anomalies or patterns
            analysis = await self._analyze_metric(metric)
            
            self.logger.info(
                "Processed metric event",
                metric_name=metric.metric_name,
                namespace=metric.namespace,
                value=metric.value
            )
            
            return {
                "status": "processed",
                "metric": metric.dict(),
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(
                "Error processing metric event",
                detail=event.get('detail'),
                error=str(e)
            )
            raise
    
    async def _determine_alarm_actions(self, alarm: CloudWatchAlarm) -> List[Dict[str, Any]]:
        """Determine actions to take based on alarm state."""
        actions = []
        
        if alarm.state_value == "ALARM":
            # Critical alarm - immediate action required
            if "CPU" in alarm.metric_name.upper():
                actions.append({
                    "type": "scale_up",
                    "resource_type": "auto_scaling_group",
                    "metric": alarm.metric_name,
                    "threshold": alarm.threshold,
                    "urgency": "high"
                })
            
            elif "Memory" in alarm.metric_name:
                actions.append({
                    "type": "investigate_memory",
                    "resource_type": "ec2_instance",
                    "metric": alarm.metric_name,
                    "urgency": "medium"
                })
            
            elif "DiskSpace" in alarm.metric_name:
                actions.append({
                    "type": "cleanup_disk",
                    "resource_type": "ec2_instance",
                    "metric": alarm.metric_name,
                    "urgency": "high"
                })
        
        elif alarm.state_value == "OK":
            # Alarm resolved - may trigger scale down
            if "CPU" in alarm.metric_name.upper():
                actions.append({
                    "type": "consider_scale_down",
                    "resource_type": "auto_scaling_group",
                    "metric": alarm.metric_name,
                    "urgency": "low"
                })
        
        return actions
    
    async def _analyze_metric(self, metric: CloudWatchMetric) -> Dict[str, Any]:
        """Analyze a metric for patterns and anomalies."""
        analysis = {
            "anomaly_detected": False,
            "trend": "stable",
            "severity": "normal",
            "recommendations": []
        }
        
        try:
            # Get historical data for comparison
            historical_data = await self._get_historical_metric_data(metric)
            
            if historical_data:
                avg_value = sum(historical_data) / len(historical_data)
                
                # Simple anomaly detection
                deviation = abs(metric.value - avg_value) / avg_value if avg_value > 0 else 0
                
                if deviation > 0.5:  # 50% deviation
                    analysis["anomaly_detected"] = True
                    analysis["severity"] = "high" if deviation > 1.0 else "medium"
                    analysis["recommendations"].append(
                        f"Metric {metric.metric_name} shows {deviation:.1%} deviation from normal"
                    )
                
                # Trend analysis
                if len(historical_data) >= 3:
                    recent_avg = sum(historical_data[-3:]) / 3
                    older_avg = sum(historical_data[:-3]) / (len(historical_data) - 3)
                    
                    if recent_avg > older_avg * 1.2:
                        analysis["trend"] = "increasing"
                    elif recent_avg < older_avg * 0.8:
                        analysis["trend"] = "decreasing"
        
        except Exception as e:
            self.logger.warning(
                "Error analyzing metric",
                metric_name=metric.metric_name,
                error=str(e)
            )
        
        return analysis
    
    async def _get_historical_metric_data(
        self, 
        metric: CloudWatchMetric, 
        hours_back: int = 24
    ) -> List[float]:
        """Get historical data for a metric."""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time.replace(hour=end_time.hour - hours_back)
            
            # Convert dimensions to CloudWatch format
            dimensions = [
                {"Name": k, "Value": v} 
                for k, v in metric.dimensions.items()
            ]
            
            response = self.cloudwatch.get_metric_statistics(
                Namespace=metric.namespace,
                MetricName=metric.metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Average']
            )
            
            return [dp['Average'] for dp in response.get('Datapoints', [])]
            
        except Exception as e:
            self.logger.warning(
                "Could not get historical metric data",
                metric_name=metric.metric_name,
                error=str(e)
            )
            return []
    
    def create_alarm(
        self,
        alarm_name: str,
        metric_name: str,
        namespace: str,
        threshold: float,
        comparison_operator: str,
        dimensions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a CloudWatch alarm.
        
        Args:
            alarm_name: Name of the alarm
            metric_name: Metric to monitor
            namespace: CloudWatch namespace
            threshold: Alarm threshold
            comparison_operator: Comparison operator
            dimensions: Metric dimensions
            
        Returns:
            Alarm creation result
        """
        try:
            alarm_dimensions = [
                {"Name": k, "Value": v}
                for k, v in (dimensions or {}).items()
            ]
            
            self.cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator=comparison_operator,
                EvaluationPeriods=2,
                MetricName=metric_name,
                Namespace=namespace,
                Period=300,
                Statistic='Average',
                Threshold=threshold,
                ActionsEnabled=True,
                AlarmDescription=f"Alarm for {metric_name}",
                Dimensions=alarm_dimensions,
                Unit='Percent' if 'Utilization' in metric_name else 'Count'
            )
            
            self.logger.info(
                "Created CloudWatch alarm",
                alarm_name=alarm_name,
                metric_name=metric_name,
                threshold=threshold
            )
            
            return {"status": "created", "alarm_name": alarm_name}
            
        except Exception as e:
            self.logger.error(
                "Failed to create CloudWatch alarm",
                alarm_name=alarm_name,
                error=str(e)
            )
            return {"status": "error", "error": str(e)}
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "processed_events": self.processed_events,
            "alarm_events": self.alarm_events,
            "metric_events": self.metric_events,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
