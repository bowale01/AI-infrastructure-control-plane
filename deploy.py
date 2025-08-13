#!/usr/bin/env python3
"""
AWS Deployment Script for AI Agents Project
Automates the deployment of the AI agents infrastructure and application.
"""

import os
import sys
import json
import boto3
import zipfile
import argparse
from pathlib import Path
from botocore.exceptions import ClientError


class DeploymentManager:
    def __init__(self, region='us-east-1', environment='development'):
        self.region = region
        self.environment = environment
        self.project_name = 'ai-agents'
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=region)
        self.cloudformation_client = boto3.client('cloudformation', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        
        # Deployment configuration
        self.bucket_name = f'{self.project_name}-deployment-{self.environment}-{region}'
        self.stack_name = f'{self.project_name}-{self.environment}'
        
    def create_deployment_bucket(self):
        """Create S3 bucket for deployment artifacts"""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            print(f"✅ Created deployment bucket: {self.bucket_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"✅ Using existing deployment bucket: {self.bucket_name}")
            else:
                print(f"❌ Error creating bucket: {e}")
                raise
    
    def package_lambda_code(self):
        """Package the source code for Lambda deployment"""
        zip_path = Path('deployment/lambda-package.zip')
        zip_path.parent.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add source code
            for root, dirs, files in os.walk('src'):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        arcname = file_path.replace('\\', '/')
                        zipf.write(file_path, arcname)
            
            # Add config files
            for config_file in ['config/agents_config.yaml', 'config/aws_config.yaml']:
                if os.path.exists(config_file):
                    zipf.write(config_file, config_file.replace('\\', '/'))
            
            # Add requirements for Lambda layer
            if os.path.exists('requirements.txt'):
                zipf.write('requirements.txt', 'requirements.txt')
        
        print(f"✅ Created Lambda package: {zip_path}")
        return zip_path
    
    def upload_artifacts(self, lambda_zip_path):
        """Upload deployment artifacts to S3"""
        try:
            # Upload Lambda package
            lambda_key = f'lambda/{self.environment}/ai-agents.zip'
            self.s3_client.upload_file(str(lambda_zip_path), self.bucket_name, lambda_key)
            lambda_url = f'https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{lambda_key}'
            print(f"✅ Uploaded Lambda package to: {lambda_url}")
            
            # Upload CloudFormation template
            cf_key = f'cloudformation/{self.environment}/main.yaml'
            self.s3_client.upload_file('infrastructure/cloudformation/main.yaml', self.bucket_name, cf_key)
            cf_url = f'https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{cf_key}'
            print(f"✅ Uploaded CloudFormation template to: {cf_url}")
            
            return lambda_url, cf_url
            
        except ClientError as e:
            print(f"❌ Error uploading artifacts: {e}")
            raise
    
    def deploy_infrastructure(self, lambda_url):
        """Deploy CloudFormation stack"""
        try:
            with open('infrastructure/cloudformation/main.yaml', 'r') as f:
                template_body = f.read()
            
            parameters = [
                {'ParameterKey': 'Environment', 'ParameterValue': self.environment},
                {'ParameterKey': 'ProjectName', 'ParameterValue': self.project_name},
            ]
            
            # Check if stack exists
            try:
                self.cloudformation_client.describe_stacks(StackName=self.stack_name)
                # Stack exists, update it
                response = self.cloudformation_client.update_stack(
                    StackName=self.stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=['CAPABILITY_NAMED_IAM']
                )
                print(f"✅ Updating CloudFormation stack: {self.stack_name}")
                
            except ClientError as e:
                if 'does not exist' in str(e):
                    # Stack doesn't exist, create it
                    response = self.cloudformation_client.create_stack(
                        StackName=self.stack_name,
                        TemplateBody=template_body,
                        Parameters=parameters,
                        Capabilities=['CAPABILITY_NAMED_IAM']
                    )
                    print(f"✅ Creating CloudFormation stack: {self.stack_name}")
                else:
                    raise
            
            # Wait for stack deployment to complete
            print("⏳ Waiting for stack deployment to complete...")
            waiter = self.cloudformation_client.get_waiter('stack_create_complete')
            waiter.wait(StackName=self.stack_name)
            
            print(f"✅ CloudFormation stack deployed successfully: {self.stack_name}")
            return True
            
        except ClientError as e:
            print(f"❌ Error deploying infrastructure: {e}")
            return False
    
    def deploy(self):
        """Main deployment workflow"""
        print(f"🚀 Starting deployment for environment: {self.environment}")
        print(f"📍 Region: {self.region}")
        print(f"🏗️  Stack: {self.stack_name}")
        
        try:
            # Step 1: Create deployment bucket
            self.create_deployment_bucket()
            
            # Step 2: Package Lambda code
            lambda_zip_path = self.package_lambda_code()
            
            # Step 3: Upload artifacts
            lambda_url, cf_url = self.upload_artifacts(lambda_zip_path)
            
            # Step 4: Deploy infrastructure
            if self.deploy_infrastructure(lambda_url):
                print("\n🎉 Deployment completed successfully!")
                print(f"📊 Monitor your deployment at: https://console.aws.amazon.com/cloudformation/home?region={self.region}#/stacks")
                print(f"🔍 View logs at: https://console.aws.amazon.com/cloudwatch/home?region={self.region}#logsV2:")
            else:
                print("\n❌ Deployment failed!")
                return False
                
        except Exception as e:
            print(f"\n❌ Deployment failed with error: {e}")
            return False
        
        return True
    
    def cleanup(self):
        """Clean up deployment resources"""
        try:
            print(f"🧹 Cleaning up resources for {self.stack_name}...")
            
            # Delete CloudFormation stack
            self.cloudformation_client.delete_stack(StackName=self.stack_name)
            print(f"✅ Initiated stack deletion: {self.stack_name}")
            
            # Wait for stack deletion
            print("⏳ Waiting for stack deletion to complete...")
            waiter = self.cloudformation_client.get_waiter('stack_delete_complete')
            waiter.wait(StackName=self.stack_name)
            
            # Clean up S3 bucket
            objects = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in objects:
                delete_keys = [{'Key': obj['Key']} for obj in objects['Contents']]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': delete_keys}
                )
            
            self.s3_client.delete_bucket(Bucket=self.bucket_name)
            print(f"✅ Deleted deployment bucket: {self.bucket_name}")
            
            print("🎉 Cleanup completed successfully!")
            
        except ClientError as e:
            print(f"❌ Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description='Deploy AI Agents to AWS')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--environment', default='development', choices=['development', 'staging', 'production'])
    parser.add_argument('--cleanup', action='store_true', help='Clean up all resources')
    
    args = parser.parse_args()
    
    deployer = DeploymentManager(region=args.region, environment=args.environment)
    
    if args.cleanup:
        deployer.cleanup()
    else:
        deployer.deploy()


if __name__ == '__main__':
    main()
