# AWS AI Agents for Infrastructure Automation

A comprehensive AI-driven automation system that leverages AWS services to provide intelligent scaling, recovery, and diagnostics through control-plane patterns and event-driven architecture.

## 🏗️ Architecture Overview

This project implements AI agents that automate infrastructure management using:

- **Control Plane Patterns**: Centralized decision-making with distributed execution
- **Event-Driven Logic**: Real-time response to infrastructure events
- **AWS Native Integration**: Leveraging AWS services for scalability and reliability

## 🚀 Features

### Core Agents
- **Scaling Agent**: Intelligent auto-scaling based on metrics and predictions
- **Recovery Agent**: Automated fault detection and recovery procedures
- **Diagnostics Agent**: Real-time monitoring and anomaly detection

### AWS Services Integration
- **AWS Lambda**: Serverless execution of agent logic
- **Amazon EventBridge**: Event routing and orchestration
- **Amazon CloudWatch**: Metrics collection and alerting
- **AWS Auto Scaling**: Dynamic resource scaling
- **AWS Systems Manager**: Configuration and patch management
- **Amazon SNS/SQS**: Messaging and notifications

## 📁 Project Structure

```
ai-infrastructure-control-plane/
├── src/
│   ├── agents/                 # Core AI agent implementations
│   │   ├── base_agent.py       # Base agent framework
│   │   └── scaling_agent.py    # Intelligent scaling agent
│   ├── control_plane/          # Control plane orchestration
│   │   └── orchestrator.py     # Multi-agent coordination
│   ├── event_handlers/         # Event processing logic
│   │   └── cloudwatch_handler.py
│   └── utils/                  # Shared utilities
│       ├── aws_client.py       # AWS SDK management
│       ├── metrics.py          # Metrics collection
│       └── logging_config.py   # Centralized logging
├── aws/                        # AWS Infrastructure Components
│   ├── cloudformation/         # Infrastructure as Code
│   │   ├── main.yaml          # Complete infrastructure stack
│   │   └── simple.yaml        # Simplified deployment
│   ├── lambda/                # Lambda functions
│   │   └── lambda_handler.py  # Serverless handler
│   ├── config/                # AWS configurations
│   │   └── aws_config.yaml    # Service settings
│   ├── deployment/            # Deployment artifacts
│   └── deploy.py              # Automated deployment
├── config/
│   └── agents_config.yaml     # Agent configurations
├── tests/                     # Test suite
└── docs/                      # Documentation
│   ├── terraform/
│   └── cdk/
├── config/                     # Configuration files
│   ├── agents_config.yaml
│   └── aws_config.yaml
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- AWS CLI configured with appropriate permissions
- AWS account with necessary service access

### Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd aws-ai-agents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install development dependencies:
```bash
pip install -e .[dev]
```

## ⚙️ Configuration

### AWS Credentials
Ensure your AWS credentials are configured:
```bash
aws configure
```

### Agent Configuration
Copy and modify the configuration files:
```bash
cp config/agents_config.yaml.example config/agents_config.yaml
cp config/aws_config.yaml.example config/aws_config.yaml
```

## 🚀 Deployment

### Using AWS CDK (Recommended)
```bash
cd infrastructure/cdk
npm install
cdk deploy
```

### Using Terraform
```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

### Using CloudFormation
```bash
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/main.yaml \
  --stack-name ai-agents-stack \
  --capabilities CAPABILITY_IAM
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📊 Monitoring and Observability

The system includes comprehensive monitoring through:
- **CloudWatch Dashboards**: Real-time metrics visualization
- **Custom Metrics**: Agent-specific performance indicators
- **Structured Logging**: Centralized log aggregation
- **Prometheus Integration**: Additional metrics collection

## 🔧 Development

### Code Style
This project uses Black for code formatting:
```bash
black src/ tests/
```

### Type Checking
Run mypy for type checking:
```bash
mypy src/
```

### Linting
Use flake8 for linting:
```bash
flake8 src/ tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 🆘 Support

For questions and support:
- Create an issue in the GitHub repository
- Check the [documentation](docs/)
- Review the [FAQ](docs/faq.md)

## 🔮 Roadmap

- [ ] Machine Learning integration for predictive scaling
- [ ] Multi-region deployment support
- [ ] Enhanced security monitoring
- [ ] Cost optimization agents
- [ ] Kubernetes integration
- [ ] Custom metric collectors
