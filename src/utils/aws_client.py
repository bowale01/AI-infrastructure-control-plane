"""
AWS Client Manager for centralized AWS service client management.
"""
import boto3
from botocore.config import Config
from typing import Dict, Any, Optional
import threading
from functools import lru_cache


class AWSClientManager:
    """
    Centralized manager for AWS service clients with connection pooling and caching.
    """
    
    def __init__(self, region_name: str = 'us-east-1', **kwargs):
        """
        Initialize the AWS Client Manager.
        
        Args:
            region_name: AWS region name
            **kwargs: Additional boto3 session parameters
        """
        self.region_name = region_name
        self.session_kwargs = kwargs
        self._clients: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Default configuration for all clients
        self.default_config = Config(
            region_name=region_name,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            },
            max_pool_connections=50
        )
    
    def get_client(self, service_name: str, **kwargs) -> Any:
        """
        Get or create an AWS service client.
        
        Args:
            service_name: Name of the AWS service
            **kwargs: Additional client configuration
            
        Returns:
            Boto3 client instance
        """
        client_key = f"{service_name}_{hash(str(sorted(kwargs.items())))}"
        
        if client_key not in self._clients:
            with self._lock:
                if client_key not in self._clients:
                    config = kwargs.pop('config', self.default_config)
                    session = boto3.Session(**self.session_kwargs)
                    
                    self._clients[client_key] = session.client(
                        service_name,
                        config=config,
                        **kwargs
                    )
        
        return self._clients[client_key]
    
    def get_resource(self, service_name: str, **kwargs) -> Any:
        """
        Get or create an AWS service resource.
        
        Args:
            service_name: Name of the AWS service
            **kwargs: Additional resource configuration
            
        Returns:
            Boto3 resource instance
        """
        resource_key = f"resource_{service_name}_{hash(str(sorted(kwargs.items())))}"
        
        if resource_key not in self._clients:
            with self._lock:
                if resource_key not in self._clients:
                    session = boto3.Session(**self.session_kwargs)
                    
                    self._clients[resource_key] = session.resource(
                        service_name,
                        region_name=self.region_name,
                        **kwargs
                    )
        
        return self._clients[resource_key]
    
    @lru_cache(maxsize=128)
    def get_account_id(self) -> str:
        """Get the current AWS account ID."""
        sts = self.get_client('sts')
        return sts.get_caller_identity()['Account']
    
    @lru_cache(maxsize=128)
    def get_caller_identity(self) -> Dict[str, str]:
        """Get the caller identity information."""
        sts = self.get_client('sts')
        return sts.get_caller_identity()
    
    def clear_cache(self) -> None:
        """Clear all cached clients and resources."""
        with self._lock:
            self._clients.clear()
            self.get_account_id.cache_clear()
            self.get_caller_identity.cache_clear()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of AWS connectivity.
        
        Returns:
            Health check results
        """
        try:
            identity = self.get_caller_identity()
            return {
                "status": "healthy",
                "account_id": identity.get("Account"),
                "user_id": identity.get("UserId"),
                "arn": identity.get("Arn"),
                "region": self.region_name,
                "client_count": len(self._clients)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "region": self.region_name,
                "client_count": len(self._clients)
            }
