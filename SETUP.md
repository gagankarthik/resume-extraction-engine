# Setup Guide

Complete instructions for local development and AWS Lambda production deployment.

---

## Prerequisites

- Python 3.11+
- An OpenAI or Anthropic API key
- (For AWS deployment) AWS CLI configured, Terraform 1.6+

---

## Local Development

### 1. Clone and install

```bash
git clone https://github.com/gagankarthik/resume-extraction-engine.git
cd resume-extraction-engine
pip install -r requirements.txt
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Provider: "openai" or "anthropic"
MODEL_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Anthropic (only needed if MODEL_PROVIDER=anthropic)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-7

# Pipeline mode: true = multi-agent, false = single-shot LLM call
USE_ORCHESTRATOR=true

# Upload size limit in MB
MAX_FILE_SIZE_MB=20
```

> **Never commit `.env` to git.** It is already in `.gitignore`.

### 3. Start the server

```bash
uvicorn main:app --reload
```

The API is available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

### 4. Test with a resume

```bash
curl -X POST "http://localhost:8000/extract" \
     -F "file=@/path/to/resume.pdf"
```

---

## Environment Variables Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| `MODEL_PROVIDER` | `openai` | No | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | If using OpenAI | Your OpenAI secret key |
| `OPENAI_MODEL` | `gpt-4o` | No | Any OpenAI chat model |
| `ANTHROPIC_API_KEY` | — | If using Anthropic | Your Anthropic secret key |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | No | Any Anthropic messages model |
| `USE_ORCHESTRATOR` | `true` | No | `true` = multi-agent pipeline (higher accuracy); `false` = single-shot (faster, cheaper) |
| `MAX_FILE_SIZE_MB` | `20` | No | Max upload size |

### Choosing a mode

| Mode | `USE_ORCHESTRATOR` | Speed | Accuracy | Cost |
|---|---|---|---|---|
| Multi-agent | `true` | 30–90s | Highest | ~8–10× LLM calls |
| Single-shot | `false` | 5–15s | Good | 1 LLM call |

Use `false` for development and testing. Use `true` for production where extraction completeness matters.

---

## AWS Lambda Deployment

The app uses [Mangum](https://mangum.faun.dev/) to wrap FastAPI for Lambda's event format. Infrastructure is managed with Terraform.

### Architecture

```
GitHub push to main
    │
    ▼
GitHub Actions
    ├─ Lint (ruff)
    ├─ Build lambda.zip (~100 MB)
    ├─ terraform init
    ├─ terraform plan
    └─ terraform apply
            │
            ▼
        AWS (us-east-2)
        ├─ S3 bucket        ← stores lambda.zip
        ├─ Lambda function  ← Python 3.11, 1024 MB, 900s timeout
        ├─ Lambda Function URL ← public HTTPS endpoint (no API Gateway)
        └─ CloudWatch logs  ← 14-day retention
```

> **Why Lambda Function URL instead of API Gateway?**
> API Gateway enforces a hard 29-second timeout. The multi-agent pipeline takes 30–90 seconds per resume. Lambda Function URLs have no such limit.

---

### Step 1 — Create the Terraform state bucket (one time only)

Terraform stores its state in S3. Create the bucket manually before the first deployment:

```bash
aws s3api create-bucket \
  --bucket resume-extraction-tfstate \
  --region us-east-2 \
  --create-bucket-configuration LocationConstraint=us-east-2

# Enable versioning on the state bucket
aws s3api put-bucket-versioning \
  --bucket resume-extraction-tfstate \
  --versioning-configuration Status=Enabled
```

---

### Step 2 — IAM permissions for the deploy user

The AWS IAM user whose keys you put in GitHub Secrets needs the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "s3:*",      "Resource": "*" },
    { "Effect": "Allow", "Action": "lambda:*",   "Resource": "*" },
    { "Effect": "Allow", "Action": "iam:*",      "Resource": "*" },
    { "Effect": "Allow", "Action": "logs:*",     "Resource": "*" }
  ]
}
```

You can attach `AdministratorAccess` for simplicity during initial setup, then tighten later.

---

### Step 3 — Add GitHub Secrets

Go to your repository → **Settings → Secrets and variables → Actions → New repository secret**.

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `MODEL_PROVIDER` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | Your OpenAI key |
| `OPENAI_MODEL` | e.g. `gpt-4o` |
| `ANTHROPIC_API_KEY` | Your Anthropic key (set to empty string if not using) |
| `ANTHROPIC_MODEL` | e.g. `claude-opus-4-7` (set to empty string if not using) |

---

### Step 4 — Deploy

Push to `main`:

```bash
git add .
git commit -m "initial deployment"
git push origin main
```

GitHub Actions will:
1. Lint the Python code with `ruff`
2. Build `lambda.zip` with all dependencies
3. Run `terraform init → plan → apply`
4. Print the **Lambda Function URL** at the end of the logs

The Function URL looks like:
```
https://xxxxxxxxxxxxxxxxxxxxxxxxxxxx.lambda-url.us-east-2.on.aws/
```

Use this URL as the API base URL in your frontend `.env`.

---

### Step 5 — Verify the deployment

```bash
# Health check
curl https://<your-function-url>/health

# Extract a resume
curl -X POST "https://<your-function-url>/extract" \
     -F "file=@resume.pdf"
```

---

## Terraform Resources Created

| Resource | Name | Details |
|---|---|---|
| IAM Role | `resume-extraction-lambda-role` | Lambda execution + S3 read |
| S3 Bucket | `resume-extraction-lambda-packages` | Stores `lambda.zip`, versioned |
| Lambda Function | `resume-extraction-engine` | Python 3.11, 1024 MB, 900s timeout |
| Lambda Function URL | — | Public HTTPS, CORS `*`, no auth |
| CloudWatch Log Group | `/aws/lambda/resume-extraction-engine` | 14-day retention |

---

## Updating the Deployment

Every push to `main` automatically rebuilds the zip and runs `terraform apply`. Terraform only replaces the Lambda function code when the zip changes (tracked by MD5 hash).

To update environment variables (e.g. change the model):

1. Update the secret in GitHub → Settings → Secrets
2. Push any commit to `main` — the next `terraform apply` will update the Lambda config

---

## Running Locally Against a Lambda Endpoint

If you want to point the frontend at your Lambda Function URL during local development:

```env
# In resume-frontend/.env.local
NEXT_PUBLIC_API_URL=https://<your-function-url>
```

---

## Troubleshooting

**Lambda times out**

The default Lambda timeout is set to 900s (15 min) by Terraform. If you see timeout errors, check CloudWatch logs — the cause is almost always the LLM provider taking longer than expected. Consider switching to `USE_ORCHESTRATOR=false` for faster single-shot extraction.

**`No readable text found`**

The uploaded PDF is likely a scanned image. `pdfplumber` requires text-based PDFs. Convert the PDF using Adobe Acrobat, Smallpdf, or run it through an OCR tool first.

**`OPENAI_API_KEY is not set`**

On Lambda, environment variables come from the function configuration, not a `.env` file. Verify the secrets are set in GitHub and were applied by the last Terraform run. Check `terraform output` in the CI logs.

**Terraform state conflict**

If two deploys run simultaneously they may conflict on the Terraform state. The S3 backend does not have DynamoDB locking by default. Avoid concurrent pushes to `main`. To add locking:

```hcl
# terraform/main.tf — add to backend block
dynamodb_table = "resume-extraction-tflock"
```

Then create the table:

```bash
aws dynamodb create-table \
  --table-name resume-extraction-tflock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-2
```
