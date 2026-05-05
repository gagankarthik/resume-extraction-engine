from mangum import Mangum
from main import app

# AWS Lambda entrypoint — use "handler.handler" as the Lambda handler setting.
# Deploy with Lambda Function URLs (not API Gateway) to avoid the 29-second
# API Gateway timeout; the multi-agent LLM pipeline takes 30-90 seconds.
# Set Lambda timeout to 900 seconds (15 min) in the function configuration.
handler = Mangum(app, lifespan="off")
