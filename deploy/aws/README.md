# AWS Deployment for VisionPilot

This directory contains deployment configurations for running VisionPilot on AWS.

## Architecture Options

### Option 1: ECS Fargate (Recommended for Contest)
- Serverless containers
- Auto-scaling
- No EC2 management
- Cost-effective for intermittent workloads

### Option 2: EC2 with Docker Compose
- Full control
- Good for GPU workloads
- Requires more management

### Option 3: Lambda (For Async Processing)
- Event-driven image processing
- Scale to zero
- Limited to 15min execution time

## ECS Fargate Deployment

### Prerequisites
- AWS CLI configured
- ECR repository created
- VPC with public/private subnets
- Application Load Balancer

### Build and Push
```bash
# Create ECR repository
aws ecr create-repository --repository-name visionpilot --region us-east-1

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t visionpilot .
docker tag visionpilot:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/visionpilot:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/visionpilot:latest
```

### ECS Service Definition
See `ecs-task-definition.json` and `ecs-service.json`

## CloudFormation Template

See `cloudformation.yaml` for complete infrastructure as code.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PYTHONPATH | Python module path | /app/src |
| LOG_LEVEL | Logging level | info |
| AWS_REGION | AWS region | us-east-1 |
| S3_BUCKET | S3 bucket for uploads | (required for S3 uploads) |

## Security

- IAM roles with least privilege
- Secrets in AWS Secrets Manager
- VPC with private subnets for containers
- ALB with WAF for protection

## Monitoring

- CloudWatch Logs for application logs
- CloudWatch Metrics for custom metrics
- X-Ray for distributed tracing (optional)