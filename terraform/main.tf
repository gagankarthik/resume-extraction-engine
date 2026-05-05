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
  handler       = "handler.handler"
  runtime       = "python3.11"
  timeout       = 900   # 15 min — multi-agent LLM pipeline takes 30-90s
  memory_size   = 1024

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
    }
  }

  depends_on = [aws_cloudwatch_log_group.api]
}

# ── Function URL (no API Gateway — avoids the 29s timeout limit) ─────────────

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = false
    allow_origins     = ["*"]
    allow_methods     = ["*"]
    allow_headers     = ["*"]
    max_age           = 86400
  }
}

# ── CloudWatch logs ──────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/resume-extraction-engine"
  retention_in_days = 14
}
