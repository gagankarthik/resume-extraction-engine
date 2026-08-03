variable "model_provider" {
  type        = string
  description = "LLM provider: openai or anthropic"
  default     = "openai"
}

variable "openai_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "openai_model" {
  type = string
  # Matches what is actually deployed. The previous default of gpt-4o disagreed
  # with production, so reading this file gave the wrong answer about the model.
  default = "gpt-4.1-mini"
}

variable "anthropic_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "anthropic_model" {
  type    = string
  default = "claude-opus-4-7"
}

variable "use_orchestrator" {
  type        = string
  description = "true = multi-agent pipeline, false = single-shot LLM call"
  default     = "true"
}

variable "extraction_shared_secret" {
  type        = string
  sensitive   = true
  description = <<-EOT
    Secret shared with the frontend, which signs a short-lived upload ticket
    with it. The engine refuses every upload without a valid ticket, so this
    must match NEXT_EXTRACTION_SHARED_SECRET in the Amplify console exactly.

    Generate with: openssl rand -hex 32

    There is deliberately no default. An empty value makes the service refuse
    all uploads rather than quietly accept anonymous ones.
  EOT
}

variable "allowed_origins" {
  type        = list(string)
  description = <<-EOT
    Origins permitted to call the engine from a browser. The upload goes
    straight from the page to the Function URL, so this is a real access
    control and not a formality — it was ["*"].

    Scheme + host + port, no trailing slash.
  EOT
  default     = ["https://hire.oceanbluecorp.com"]
}
