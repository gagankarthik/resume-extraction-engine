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
  type    = string
  default = "gpt-4o"
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
