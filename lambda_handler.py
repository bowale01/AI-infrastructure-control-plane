"""
Simplified Lambda handler for AI Agents.
This version avoids complex dependencies for easier deployment.
"""
import json
import logging
import boto3
from datetime import datetime

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    AWS Lambda handler for the AI Agents system.
    
    Args:
        event: Lambda event data
        context: Lambda context object
        
    Returns:
        Dict: Response with status and results
    """
    try:
        logger.info(f"AI Agents Lambda triggered with event: {json.dumps(event)}")
        
        # Initialize AWS clients
        cloudwatch = boto3.client('cloudwatch')
        
        # Get some basic CloudWatch metrics as a demonstration
        try:
            # Get EC2 CPU utilization for the last hour
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[],
                StartTime=datetime.utcnow().replace(hour=datetime.utcnow().hour-1),
                EndTime=datetime.utcnow(),
                Period=300,
                Statistics=['Average', 'Maximum']
            )
            
            metrics_data = response.get('Datapoints', [])
            logger.info(f"Retrieved {len(metrics_data)} CloudWatch datapoints")
            
        except Exception as metrics_error:
            logger.warning(f"Failed to retrieve CloudWatch metrics: {metrics_error}")
            metrics_data = []
        
        # Prepare response
        response_body = {
            'message': 'AI Agents system executed successfully',
            'timestamp': datetime.utcnow().isoformat(),
            'function_name': context.function_name,
            'request_id': context.aws_request_id,
            'event_processed': event,
            'environment': 'development',
            'metrics_retrieved': len(metrics_data),
            'status': 'healthy'
        }
        
        # Add some sample scaling logic demonstration
        if event.get('action') == 'check_scaling':
            response_body['scaling_analysis'] = {
                'current_metrics': metrics_data[-1] if metrics_data else None,
                'recommendation': 'Monitor for 5 more minutes',
                'action_taken': 'none',
                'reason': 'Insufficient data or stable metrics'
            }
        
        response = {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }
        
        logger.info(f"AI Agents execution completed successfully")
        return response
        
    except Exception as e:
        error_msg = f"AI Agents execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'AI Agents execution failed',
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': context.aws_request_id
            })
        }
