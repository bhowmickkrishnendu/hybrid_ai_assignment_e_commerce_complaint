# AWS Deployment Notes

## Deployment Option: AWS App Runner

AWS App Runner is the easiest way to deploy this FastAPI app. It handles containers, scaling, and HTTPS automatically.

---

## Prerequisites

1. AWS account with appropriate IAM permissions
2. AWS CLI installed and configured (`aws configure`)
3. Docker installed locally
4. Models trained and saved (`.pkl` and `.keras` files)

---

## Step 1: Push Docker Image to AWS ECR

```bash
(Bash Users)

# Set your variables
AWS_ACCOUNT_ID=123456789012
AWS_REGION=us-east-1
REPO_NAME=hybrid-ai-app

# Create ECR repository
aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and tag image
docker build -t $REPO_NAME .
docker tag $REPO_NAME:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest
```

```powershell
(Powershell Users)

# Set variables
$env:AWS_ACCOUNT_ID = "123456789012"
$env:AWS_REGION = "us-east-1"
$env:REPO_NAME = "hybrid-ai-app"

# Create ECR repository
aws ecr create-repository `
  --repository-name $env:REPO_NAME `
  --region $env:AWS_REGION

# ECR URL
$ECR_URL = "$($env:AWS_ACCOUNT_ID).dkr.ecr.$($env:AWS_REGION).amazonaws.com"

# Authenticate Docker to ECR
aws ecr get-login-password --region $env:AWS_REGION `
| docker login --username AWS --password-stdin $ECR_URL

# Build Docker image
docker build -t $env:REPO_NAME .

# Tag image
docker tag "${env:REPO_NAME}:latest" `
  "$ECR_URL/$($env:REPO_NAME):latest"

# Push image to ECR
docker push "$ECR_URL/$($env:REPO_NAME):latest"
```

---

## Step 2: Create App Runner Service

```bash
(Bash Users)

# Create App Runner service
aws apprunner create-service \
  --service-name hybrid-ai-complaint \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "123456789012.dkr.ecr.us-east-1.amazonaws.com/hybrid-ai-app:latest",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {
          "GEMINI_API_KEY": "your_gemini_key_here"
        }
      },
      "ImageRepositoryType": "ECR"
    },
    "AutoDeploymentsEnabled": false
  }' \
  --instance-configuration '{"Cpu": "1 vCPU", "Memory": "2 GB"}'
```

```powershell
(Powershell Users)

# App Runner service creation

$IMAGE_IDENTIFIER = "$($env:AWS_ACCOUNT_ID).dkr.ecr.$($env:AWS_REGION).amazonaws.com/$($env:REPO_NAME):latest"

aws apprunner create-service `
  --service-name "hybrid-ai-complaint" `
  --source-configuration "{
    `"ImageRepository`": {
      `"ImageIdentifier`": `"$IMAGE_IDENTIFIER`",
      `"ImageConfiguration`": {
        `"Port`": `"8000`",
        `"RuntimeEnvironmentVariables`": {
          `"GEMINI_API_KEY`": `"your_gemini_key_here`"
        }
      },
      `"ImageRepositoryType`": `"ECR`"
    },
    `"AutoDeploymentsEnabled`": false
  }" `
  --instance-configuration "{
    `"Cpu`": `"1 vCPU`",
    `"Memory`": `"2 GB`"
  }"

```

---

## Step 3: Test the Deployed Endpoint

```bash
(Bash Users)

# Get your App Runner URL from the console or:
SERVICE_URL=https://xxxxxxxxxxxx.us-east-1.awsapprunner.com

# Health check
curl $SERVICE_URL/health

# Sample prediction request
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C001",
    "complaint_text": "Product broke on day one. Support has not replied.",
    "customer_tenure_years": 0.5,
    "previous_complaints": 3,
    "order_value": 149.99,
    "account_type": "standard",
    "image_category": "damaged_product"
  }'
```
```powershell
(Powershell Users)

# Get App Runner service URL
$SERVICE_URL = aws apprunner list-services `
  --region $env:AWS_REGION `
  --query "ServiceSummaryList[?ServiceName=='hybrid-ai-complaint'].ServiceUrl" `
  --output text

# Print URL
echo "Service URL: https://$SERVICE_URL"

# Health check
curl "https://$SERVICE_URL/health"

# Sample prediction request
curl -X POST "https://$SERVICE_URL/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "customer_id": "C001",
    "complaint_text": "Product broke on day one. Support has not replied.",
    "customer_tenure_years": 0.5,
    "previous_complaints": 3,
    "order_value": 149.99,
    "account_type": "standard",
    "image_category": "damaged_product"
  }'
```

---

## Sample Request

```json
POST /predict
Content-Type: application/json

{
  "customer_id": "C001",
  "complaint_text": "Product broke on day one. Nobody answers my emails.",
  "customer_tenure_years": 0.5,
  "previous_complaints": 3,
  "order_value": 149.99,
  "account_type": "standard",
  "image_category": "damaged_product"
}
```

## Sample Response

```json
{
  "customer_id": "C001",
  "model_outputs": {
    "classical_score": 0.78,
    "cnn_score": 0.70,
    "rnn_score": 0.85,
    "final_score": 0.78,
    "risk_level": "high"
  },
  "llm_explanation": {
    "summary": "Customer reports broken product with no support response.",
    "recommended_action": "escalate_immediately",
    "reason": "High risk score from multiple escalation signals.",
    "human_review_required": true,
    "risk_notes": ["Product damage mentioned", "Support non-responsiveness indicated"]
  },
  "decision": {
    "final_action": "escalate_immediately",
    "human_required": true,
    "model_disagree": false,
    "score_range": 0.15
  }
}
```

---

## Estimated Latency

| Component | Latency |
|-----------|---------|
| Classical ML | ~10ms |
| CNN | ~50ms |
| RNN/LSTM | ~30ms |
| LLM (Gemini) | ~1–3s |
| Total end-to-end | ~1.5–4s |

---

## API Key Security

- **Never** hardcode API keys in your code or Dockerfile.
- Use AWS App Runner environment variables or AWS Secrets Manager.
- Rotate API keys regularly.
- In production, use IAM roles and Vertex AI instead of plain API keys.

---

## Cleanup (IMPORTANT — Avoid AWS Charges)

```bash
# Delete App Runner service
aws apprunner delete-service --service-arn arn:aws:apprunner:us-east-1:123456789012:service/hybrid-ai-complaint/xxxx

# Delete ECR repository (removes all images)
aws ecr delete-repository --repository-name hybrid-ai-app --force --region us-east-1

# Confirm: check AWS Console → App Runner and ECR are empty
# Estimated cost for this assignment demo: < $1 USD if deleted within a few hours
```

---

## Cost Notes

| Resource | Estimated Cost |
|----------|----------------|
| App Runner (1 vCPU / 2GB, 1 hour) | ~$0.06 |
| ECR storage (< 1 GB) | ~$0.01 |
| Gemini API calls (100 requests) | ~$0.00 (free tier) |
| **Total (demo, 1 hour)** | **< $0.10 USD** |

Delete all resources after grading to avoid ongoing charges.
