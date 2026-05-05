output "function_url" {
  description = "Lambda Function URL — use this as your API endpoint in the frontend"
  value       = aws_lambda_function_url.api.function_url
}

output "function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.api.arn
}
