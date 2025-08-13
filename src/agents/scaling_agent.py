"""
Scaling Agent for intelligent auto-scaling based on metrics and predictions.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentConfig
from ..utils.aws_client import AWSClientManager


class ScalingRule(BaseModel):
    """Configuration for a scaling rule."""
    
    metric_name: str
    threshold: float
    comparison: str = Field(pattern="^(>|<|>=|<=|==)$")
    scale_direction: str = Field(pattern="^(up|down)$")
    scale_amount: int = Field(gt=0)
    cooldown_period: int = Field(default=300, description="Cooldown period in seconds")


class ScalingTarget(BaseModel):
    """Target resource for scaling operations."""
    
    resource_type: str  # "ec2", "ecs", "lambda", etc.
    resource_id: str
    min_capacity: int = Field(ge=0)
    max_capacity: int = Field(gt=0)
    current_capacity: Optional[int] = None
    scaling_rules: List[ScalingRule] = Field(default_factory=list)
    last_scaling_action: Optional[datetime] = None


class ScalingAgent(BaseAgent):
    """
    AI agent for intelligent auto-scaling based on metrics and predictions.
    """
    
    def __init__(self, config: AgentConfig, aws_client_manager: AWSClientManager):
        """
        Initialize the scaling agent.
        
        Args:
            config: Agent configuration
            aws_client_manager: AWS client manager instance
        """
        super().__init__(config, aws_client_manager)
        self.scaling_targets: List[ScalingTarget] = []
        self.scaling_history: List[Dict[str, Any]] = []
        
        # AWS service clients
        self.cloudwatch = self.aws_clients.get_client('cloudwatch')
        self.autoscaling = self.aws_clients.get_client('autoscaling')
        self.ecs = self.aws_clients.get_client('ecs')
        self.lambda_client = self.aws_clients.get_client('lambda')
    
    async def analyze(self) -> Dict[str, Any]:
        """
        Analyze current metrics and determine scaling needs.
        
        Returns:
            Analysis results containing metrics and scaling recommendations
        """
        analysis_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "scaling_recommendations": [],
            "anomalies_detected": []
        }
        
        try:
            # Get current metrics for all scaling targets
            for target in self.scaling_targets:
                target_metrics = await self._collect_target_metrics(target)
                analysis_results["metrics"][target.resource_id] = target_metrics
                
                # Analyze each scaling rule
                for rule in target.scaling_rules:
                    recommendation = await self._analyze_scaling_rule(target, rule, target_metrics)
                    if recommendation:
                        analysis_results["scaling_recommendations"].append(recommendation)
                
                # Detect anomalies using statistical analysis
                anomalies = await self._detect_anomalies(target, target_metrics)
                if anomalies:
                    analysis_results["anomalies_detected"].extend(anomalies)
            
            # Predictive analysis based on historical data
            predictions = await self._predict_future_load()
            analysis_results["predictions"] = predictions
            
            self.logger.info(
                "Scaling analysis completed",
                targets_analyzed=len(self.scaling_targets),
                recommendations=len(analysis_results["scaling_recommendations"]),
                anomalies=len(analysis_results["anomalies_detected"])
            )
            
        except Exception as e:
            self.logger.error("Error during scaling analysis", error=str(e), exc_info=True)
            raise
        
        return analysis_results
    
    async def decide(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make scaling decisions based on analysis results.
        
        Args:
            analysis: Results from the analyze phase
            
        Returns:
            Scaling decisions to execute
        """
        decisions = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
            "deferred_actions": [],
            "rejected_actions": []
        }
        
        try:
            # Process scaling recommendations
            for recommendation in analysis.get("scaling_recommendations", []):
                decision = await self._evaluate_scaling_recommendation(recommendation, analysis)
                
                if decision["approved"]:
                    decisions["actions"].append(decision["action"])
                elif decision["deferred"]:
                    decisions["deferred_actions"].append(decision)
                else:
                    decisions["rejected_actions"].append(decision)
            
            # Handle anomaly-based decisions
            for anomaly in analysis.get("anomalies_detected", []):
                anomaly_decision = await self._handle_anomaly(anomaly, analysis)
                if anomaly_decision:
                    decisions["actions"].append(anomaly_decision)
            
            # Apply predictive scaling decisions
            predictive_actions = await self._apply_predictive_scaling(analysis.get("predictions", {}))
            decisions["actions"].extend(predictive_actions)
            
            # Prioritize and validate actions
            decisions["actions"] = await self._prioritize_scaling_actions(decisions["actions"])
            
            self.logger.info(
                "Scaling decisions made",
                actions=len(decisions["actions"]),
                deferred=len(decisions["deferred_actions"]),
                rejected=len(decisions["rejected_actions"])
            )
            
        except Exception as e:
            self.logger.error("Error during scaling decision making", error=str(e), exc_info=True)
            raise
        
        return decisions
    
    async def execute(self, decisions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute scaling decisions.
        
        Args:
            decisions: Scaling decisions to execute
            
        Returns:
            Execution results
        """
        execution_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "successful_actions": [],
            "failed_actions": [],
            "skipped_actions": []
        }
        
        try:
            for action in decisions.get("actions", []):
                try:
                    # Execute scaling action
                    result = await self._execute_scaling_action(action)
                    
                    if result["success"]:
                        execution_results["successful_actions"].append(result)
                        
                        # Update scaling history
                        self.scaling_history.append({
                            "timestamp": datetime.now(timezone.utc),
                            "action": action,
                            "result": result
                        })
                        
                        # Update target's last scaling action time
                        await self._update_target_scaling_time(action["target_id"])
                        
                    else:
                        execution_results["failed_actions"].append(result)
                        
                except Exception as e:
                    error_result = {
                        "action": action,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    execution_results["failed_actions"].append(error_result)
                    self.logger.error(
                        "Failed to execute scaling action",
                        action=action,
                        error=str(e),
                        exc_info=True
                    )
            
            self.logger.info(
                "Scaling execution completed",
                successful=len(execution_results["successful_actions"]),
                failed=len(execution_results["failed_actions"]),
                skipped=len(execution_results["skipped_actions"])
            )
            
        except Exception as e:
            self.logger.error("Error during scaling execution", error=str(e), exc_info=True)
            raise
        
        return execution_results
    
    async def _collect_target_metrics(self, target: ScalingTarget) -> Dict[str, Any]:
        """Collect current metrics for a scaling target."""
        metrics = {}
        
        try:
            # Get CloudWatch metrics
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=10)
            
            # Common metrics for different resource types
            metric_queries = self._get_metric_queries_for_target(target)
            
            for metric_name, query in metric_queries.items():
                response = self.cloudwatch.get_metric_statistics(
                    Namespace=query["namespace"],
                    MetricName=query["metric_name"],
                    Dimensions=query["dimensions"],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,
                    Statistics=['Average', 'Maximum']
                )
                
                if response['Datapoints']:
                    latest_datapoint = max(response['Datapoints'], key=lambda x: x['Timestamp'])
                    metrics[metric_name] = {
                        "average": latest_datapoint['Average'],
                        "maximum": latest_datapoint['Maximum'],
                        "timestamp": latest_datapoint['Timestamp']
                    }
                else:
                    metrics[metric_name] = {"average": 0, "maximum": 0, "timestamp": None}
            
            # Get current capacity
            current_capacity = await self._get_current_capacity(target)
            metrics["current_capacity"] = current_capacity
            target.current_capacity = current_capacity
            
        except Exception as e:
            self.logger.error(
                "Error collecting metrics for target",
                target_id=target.resource_id,
                error=str(e)
            )
            raise
        
        return metrics
    
    def _get_metric_queries_for_target(self, target: ScalingTarget) -> Dict[str, Dict[str, Any]]:
        """Get appropriate metric queries based on target type."""
        queries = {}
        
        if target.resource_type == "ec2":
            queries = {
                "cpu_utilization": {
                    "namespace": "AWS/EC2",
                    "metric_name": "CPUUtilization",
                    "dimensions": [{"Name": "InstanceId", "Value": target.resource_id}]
                },
                "network_in": {
                    "namespace": "AWS/EC2",
                    "metric_name": "NetworkIn",
                    "dimensions": [{"Name": "InstanceId", "Value": target.resource_id}]
                }
            }
        elif target.resource_type == "ecs":
            queries = {
                "cpu_utilization": {
                    "namespace": "AWS/ECS",
                    "metric_name": "CPUUtilization",
                    "dimensions": [{"Name": "ServiceName", "Value": target.resource_id}]
                },
                "memory_utilization": {
                    "namespace": "AWS/ECS",
                    "metric_name": "MemoryUtilization",
                    "dimensions": [{"Name": "ServiceName", "Value": target.resource_id}]
                }
            }
        elif target.resource_type == "lambda":
            queries = {
                "duration": {
                    "namespace": "AWS/Lambda",
                    "metric_name": "Duration",
                    "dimensions": [{"Name": "FunctionName", "Value": target.resource_id}]
                },
                "invocations": {
                    "namespace": "AWS/Lambda",
                    "metric_name": "Invocations",
                    "dimensions": [{"Name": "FunctionName", "Value": target.resource_id}]
                }
            }
        
        return queries
    
    async def _get_current_capacity(self, target: ScalingTarget) -> int:
        """Get current capacity for a scaling target."""
        try:
            if target.resource_type == "ec2":
                # For EC2, we'll use Auto Scaling Groups
                response = self.autoscaling.describe_auto_scaling_groups(
                    AutoScalingGroupNames=[target.resource_id]
                )
                if response['AutoScalingGroups']:
                    return response['AutoScalingGroups'][0]['DesiredCapacity']
                    
            elif target.resource_type == "ecs":
                # For ECS services
                response = self.ecs.describe_services(
                    services=[target.resource_id]
                )
                if response['services']:
                    return response['services'][0]['desiredCount']
                    
            elif target.resource_type == "lambda":
                # For Lambda, we'll use provisioned concurrency
                response = self.lambda_client.get_provisioned_concurrency_config(
                    FunctionName=target.resource_id
                )
                return response['AllocatedConcurrency']
                
        except Exception as e:
            self.logger.warning(
                "Could not get current capacity",
                target_id=target.resource_id,
                error=str(e)
            )
            
        return target.current_capacity or 1
    
    async def _analyze_scaling_rule(
        self, 
        target: ScalingTarget, 
        rule: ScalingRule, 
        metrics: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze a scaling rule against current metrics."""
        if rule.metric_name not in metrics:
            return None
        
        metric_value = metrics[rule.metric_name].get("average", 0)
        threshold_met = self._evaluate_threshold(metric_value, rule.threshold, rule.comparison)
        
        if threshold_met:
            # Check cooldown period
            if target.last_scaling_action:
                time_since_last = datetime.now(timezone.utc) - target.last_scaling_action
                if time_since_last.total_seconds() < rule.cooldown_period:
                    return None
            
            return {
                "type": "rule_based",
                "target_id": target.resource_id,
                "rule": rule.dict(),
                "current_value": metric_value,
                "threshold": rule.threshold,
                "scale_direction": rule.scale_direction,
                "scale_amount": rule.scale_amount,
                "confidence": 0.8  # Rule-based has high confidence
            }
        
        return None
    
    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate if a value meets the threshold condition."""
        if comparison == ">":
            return value > threshold
        elif comparison == "<":
            return value < threshold
        elif comparison == ">=":
            return value >= threshold
        elif comparison == "<=":
            return value <= threshold
        elif comparison == "==":
            return abs(value - threshold) < 0.01
        return False
    
    async def _detect_anomalies(
        self, 
        target: ScalingTarget, 
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics using statistical analysis."""
        anomalies = []
        
        # Simple anomaly detection based on historical data
        # In a real implementation, you would use more sophisticated algorithms
        
        for metric_name, metric_data in metrics.items():
            if metric_name == "current_capacity":
                continue
                
            current_value = metric_data.get("average", 0)
            
            # Get historical data for comparison
            historical_avg = await self._get_historical_average(target, metric_name)
            
            if historical_avg > 0:
                deviation = abs(current_value - historical_avg) / historical_avg
                
                if deviation > 0.5:  # 50% deviation threshold
                    anomalies.append({
                        "type": "statistical_anomaly",
                        "target_id": target.resource_id,
                        "metric_name": metric_name,
                        "current_value": current_value,
                        "historical_average": historical_avg,
                        "deviation_percentage": deviation * 100,
                        "severity": "high" if deviation > 1.0 else "medium"
                    })
        
        return anomalies
    
    async def _get_historical_average(self, target: ScalingTarget, metric_name: str) -> float:
        """Get historical average for a metric (simplified implementation)."""
        # In a real implementation, you would query historical data from CloudWatch
        # or maintain your own historical metrics database
        return 50.0  # Placeholder value
    
    async def _predict_future_load(self) -> Dict[str, Any]:
        """Predict future load based on historical patterns."""
        # Simplified predictive analysis
        # In a real implementation, you would use machine learning models
        
        predictions = {
            "next_hour": {"load_increase_probability": 0.3, "confidence": 0.6},
            "next_4_hours": {"load_increase_probability": 0.7, "confidence": 0.5},
            "peak_hours": ["09:00", "13:00", "17:00"]
        }
        
        return predictions
    
    async def _evaluate_scaling_recommendation(
        self, 
        recommendation: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate and approve/reject scaling recommendations."""
        target_id = recommendation["target_id"]
        target = next((t for t in self.scaling_targets if t.resource_id == target_id), None)
        
        if not target:
            return {"approved": False, "deferred": False, "reason": "Target not found"}
        
        current_capacity = target.current_capacity or 1
        scale_direction = recommendation["scale_direction"]
        scale_amount = recommendation["scale_amount"]
        
        if scale_direction == "up":
            new_capacity = current_capacity + scale_amount
            if new_capacity > target.max_capacity:
                return {
                    "approved": False,
                    "deferred": False,
                    "reason": f"Would exceed max capacity ({target.max_capacity})"
                }
        else:  # scale down
            new_capacity = current_capacity - scale_amount
            if new_capacity < target.min_capacity:
                return {
                    "approved": False,
                    "deferred": False,
                    "reason": f"Would go below min capacity ({target.min_capacity})"
                }
        
        # Check if there are conflicting recommendations
        conflicting = any(
            r["target_id"] == target_id and r["scale_direction"] != scale_direction
            for r in analysis.get("scaling_recommendations", [])
            if r != recommendation
        )
        
        if conflicting:
            return {
                "approved": False,
                "deferred": True,
                "reason": "Conflicting scaling recommendations detected"
            }
        
        return {
            "approved": True,
            "deferred": False,
            "action": {
                "type": "scale",
                "target_id": target_id,
                "resource_type": target.resource_type,
                "current_capacity": current_capacity,
                "new_capacity": new_capacity,
                "scale_direction": scale_direction,
                "scale_amount": scale_amount,
                "reason": recommendation.get("type", "unknown"),
                "confidence": recommendation.get("confidence", 0.5)
            }
        }
    
    async def _handle_anomaly(
        self, 
        anomaly: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle detected anomalies with appropriate scaling actions."""
        if anomaly["severity"] == "high":
            # For high severity anomalies, suggest scaling up
            return {
                "type": "scale",
                "target_id": anomaly["target_id"],
                "scale_direction": "up",
                "scale_amount": 2,  # Conservative scaling
                "reason": "anomaly_response",
                "confidence": 0.6,
                "anomaly_details": anomaly
            }
        
        return None
    
    async def _apply_predictive_scaling(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply predictive scaling based on load predictions."""
        actions = []
        
        # Simple predictive scaling logic
        next_hour_prob = predictions.get("next_hour", {}).get("load_increase_probability", 0)
        
        if next_hour_prob > 0.7:  # High probability of increased load
            for target in self.scaling_targets:
                if target.current_capacity and target.current_capacity < target.max_capacity:
                    actions.append({
                        "type": "scale",
                        "target_id": target.resource_id,
                        "resource_type": target.resource_type,
                        "current_capacity": target.current_capacity,
                        "new_capacity": min(target.current_capacity + 1, target.max_capacity),
                        "scale_direction": "up",
                        "scale_amount": 1,
                        "reason": "predictive_scaling",
                        "confidence": predictions.get("next_hour", {}).get("confidence", 0.5)
                    })
        
        return actions
    
    async def _prioritize_scaling_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize scaling actions based on confidence and urgency."""
        # Sort by confidence (descending) and then by urgency
        prioritized = sorted(
            actions,
            key=lambda x: (x.get("confidence", 0), x.get("urgency", 0)),
            reverse=True
        )
        
        return prioritized
    
    async def _execute_scaling_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific scaling action."""
        result = {
            "action": action,
            "success": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {}
        }
        
        try:
            resource_type = action["resource_type"]
            target_id = action["target_id"]
            new_capacity = action["new_capacity"]
            
            if resource_type == "ec2":
                # Scale Auto Scaling Group
                response = self.autoscaling.update_auto_scaling_group(
                    AutoScalingGroupName=target_id,
                    DesiredCapacity=new_capacity
                )
                result["details"]["autoscaling_response"] = response
                
            elif resource_type == "ecs":
                # Scale ECS Service
                response = self.ecs.update_service(
                    service=target_id,
                    desiredCount=new_capacity
                )
                result["details"]["ecs_response"] = response
                
            elif resource_type == "lambda":
                # Update Lambda provisioned concurrency
                response = self.lambda_client.put_provisioned_concurrency_config(
                    FunctionName=target_id,
                    ProvisionedConcurrencyConfig={
                        'AllocatedConcurrency': new_capacity
                    }
                )
                result["details"]["lambda_response"] = response
            
            result["success"] = True
            
            self.logger.info(
                "Scaling action executed successfully",
                target_id=target_id,
                resource_type=resource_type,
                new_capacity=new_capacity
            )
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(
                "Failed to execute scaling action",
                action=action,
                error=str(e),
                exc_info=True
            )
        
        return result
    
    async def _update_target_scaling_time(self, target_id: str) -> None:
        """Update the last scaling action time for a target."""
        for target in self.scaling_targets:
            if target.resource_id == target_id:
                target.last_scaling_action = datetime.now(timezone.utc)
                break
    
    def add_scaling_target(self, target: ScalingTarget) -> None:
        """Add a new scaling target."""
        self.scaling_targets.append(target)
        self.logger.info("Added scaling target", target_id=target.resource_id)
    
    def remove_scaling_target(self, target_id: str) -> bool:
        """Remove a scaling target."""
        original_count = len(self.scaling_targets)
        self.scaling_targets = [t for t in self.scaling_targets if t.resource_id != target_id]
        
        if len(self.scaling_targets) < original_count:
            self.logger.info("Removed scaling target", target_id=target_id)
            return True
        
        return False
    
    def get_scaling_targets(self) -> List[ScalingTarget]:
        """Get all scaling targets."""
        return self.scaling_targets.copy()
    
    def get_scaling_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get scaling history."""
        return self.scaling_history[-limit:]
