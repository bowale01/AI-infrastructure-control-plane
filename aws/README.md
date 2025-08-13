# AWS Infrastructure Components

This directory contains all AWS-specific components for the AI Infrastructure Control Plane.

## Directory Structure

### 📁 `cloudformation/`
CloudFormation templates for infrastructure deployment:
- `main.yaml` - Complete infrastructure stack with all AWS services
- `simple.yaml` - Simplified stack for basic deployment

### 📁 `config/`
AWS-specific configuration files:
- `aws_config.yaml` - AWS service configurations, regions, and settings

### 📁 `lambda/`
AWS Lambda function code and handlers:
- `lambda_handler.py` - Simplified Lambda handler for deployment

### 📁 `deployment/`
Deployment artifacts and packages:
- `*.zip` files - Lambda deployment packages
- Automated deployment scripts output

### 📄 `deploy.py`
Main deployment automation script for AWS infrastructure:
- Creates S3 deployment buckets
- Packages Lambda functions
- Deploys CloudFormation stacks
- Manages stack lifecycle

## Usage

Deploy the complete infrastructure:
```bash
python aws/deploy.py --region us-east-1 --environment production
```

Deploy using CloudFormation directly:
```bash
aws cloudformation create-stack \
  --stack-name ai-agents-prod \
  --template-body file://aws/cloudformation/main.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## Architecture

The AWS components implement:
- **Serverless Computing**: Lambda functions for agent execution
- **Infrastructure as Code**: CloudFormation for reproducible deployments
- **Event-Driven Architecture**: CloudWatch events and metrics integration
- **Security**: IAM roles and policies for least privilege access
- **Monitoring**: CloudWatch logs and metrics collection
