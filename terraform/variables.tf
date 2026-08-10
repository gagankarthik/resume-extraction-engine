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

    Every hostname that serves the frontend has to be listed, or uploads from
    the ones left out fail preflight. Amplify serves this app from three: the
    custom domain, its www subdomain, and the branch's default amplifyapp.com
    address, which stays reachable whether or not anyone means to use it.

    Scheme + host + port, no trailing slash.
  EOT
  default = [
    "https://hire.oceanbluecorp.com",
    "https://www.hire.oceanbluecorp.com",
    "https://master.dh8jqvx96jdpd.amplifyapp.com",
  ]
}
