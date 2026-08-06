terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state — create this S3 bucket ONCE manually before first apply:
  #   aws s3api create-bucket --bucket resume-extraction-tfstate \
  #     --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2
  backend "s3" {
    bucket = "resume-extraction-tfstate"
    key    = "lambda/terraform.tfstate"
    region = "us-east-2"
  }
}

provider "aws" {
  region = "us-east-2"
}

# ── IAM ─────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "lambda" {
  name = "resume-extraction-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3" {
  name = "lambda-read-package-bucket"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject"]
      Resource = "${aws_s3_bucket.packages.arn}/*"
    }]
  })
}

# ── S3 bucket for Lambda zip ─────────────────────────────────────────────────

resource "aws_s3_bucket" "packages" {
  bucket = "resume-extraction-lambda-packages"
}

resource "aws_s3_bucket_versioning" "packages" {
  bucket = aws_s3_bucket.packages.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "packages" {
  bucket                  = aws_s3_bucket.packages.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# CI uploads lambda.zip to this key before running terraform apply
resource "aws_s3_object" "zip" {
  bucket = aws_s3_bucket.packages.id
  key    = "lambda.zip"
  source = "${path.module}/../lambda.zip"
  etag   = filemd5("${path.module}/../lambda.zip")
}

# ── Lambda function ──────────────────────────────────────────────────────────

resource "aws_lambda_function" "api" {
  function_name = "resume-extraction-engine"
  role          = aws_iam_role.lambda.arn
  # Shown in the Lambda console list, where several functions look alike.
  description = "Resume extraction engine — State Format Tool (multi-agent LLM pipeline). Called by the frontend at hire.oceanbluecorp.com."
  handler     = "handler.handler"
  runtime     = "python3.11"
  timeout = 300 # 5 min — the pipeline budgets itself to 150s; this is the backstop
  # The pipeline is I/O-bound on the model API, but Lambda scales CPU with
  # memory, and CPU is what runs the event loop driving a dozen concurrent
  # calls plus the deterministic audit passes over the whole document. At 1024
  # MB that loop was itself a bottleneck.
  memory_size = 2048

  s3_bucket        = aws_s3_bucket.packages.id
  s3_key           = aws_s3_object.zip.key
  source_code_hash = filebase64sha256("${path.module}/../lambda.zip")

  environment {
    variables = {
      MODEL_PROVIDER    = var.model_provider
      OPENAI_API_KEY    = var.openai_api_key
      OPENAI_MODEL      = var.openai_model
      ANTHROPIC_API_KEY = var.anthropic_api_key
      ANTHROPIC_MODEL   = var.anthropic_model
      USE_ORCHESTRATOR  = var.use_orchestrator
      MAX_FILE_SIZE_MB  = "20"

      # What the person who uploaded the file will actually wait. This used to
      # be 840s, which is not a wait anyone sits through — it was raised to that
      # because dense resumes were measuring 624s, and the reason they measured
      # 624s was that LLM_MAX_CONCURRENT defaulted to 2, turning every parallel
      # stage into a queue two deep. With that fixed the same resumes finish in
      # well under a minute, and this is now a budget the pipeline spends down:
      # refinement stages drop out as it runs low and the resume still returns.
      EXTRACTION_TIMEOUT_SECONDS = "150"

      # In-flight model calls. The value that makes the pipeline parallel in
      # fact and not just in shape.
      LLM_MAX_CONCURRENT = "12"

      # No single call may stall the run. The SDK default is ten minutes.
      LLM_CALL_TIMEOUT_SECONDS = "90"

      # Uploads are refused without this. It is the only thing between a public
      # Function URL and an open ten-agent GPT pipeline — see auth.py.
      EXTRACTION_SHARED_SECRET = var.extraction_shared_secret

      # Mirrors the Function URL cors block below, so a local or non-Function-URL
      # run of the same image enforces the same origins.
      ALLOWED_ORIGINS = join(",", var.allowed_origins)
    }
  }

  depends_on = [aws_cloudwatch_log_group.api]
}

# ── Function URL (no API Gateway — avoids the 29s timeout limit) ─────────────
#
# The browser posts resumes straight here. Anything that proxies this call —
# API Gateway at 29s, or Amplify's CloudFront at 30s — cuts off a pipeline that
# legitimately runs 30-90s and returns a 504, so the long-lived Function URL is
# the point rather than an optimisation.
#
# authorization_type stays NONE because a browser cannot SigV4-sign an upload.
# That makes AWS's layer wide open by design, and moves the whole burden of
# authorisation onto the app: the X-Extraction-Token check in auth.py, plus the
# origin allowlist below. Neither is optional.

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = false
    # Was ["*"], which let any page on the internet spend the OpenAI budget.
    allow_origins = var.allowed_origins
    # POST only. "OPTIONS" is rejected by the API — each allowMethods member is
    # capped at six characters, and preflight is answered by the Function URL
    # itself, so listing it is both invalid and unnecessary.
    allow_methods = ["POST"]
    allow_headers = ["content-type", "x-extraction-token"]
    max_age       = 86400
  }
}

# ── CloudWatch logs ──────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/resume-extraction-engine"
  retention_in_days = 14
}
