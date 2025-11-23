"""
Amazon Bedrock client with retry logic and error handling.
"""

import json
import logging
from typing import Any, Dict, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from .config import settings

logger = logging.getLogger(__name__)


class BedrockClientError(Exception):
    """Custom exception for Bedrock client errors."""

    pass


class BedrockTimeoutError(BedrockClientError):
    """Exception for Bedrock timeout errors."""

    pass


class BedrockRateLimitError(BedrockClientError):
    """Exception for Bedrock rate limit errors."""

    pass


class BedrockClient:
    """
    Amazon Bedrock client with retry logic and error handling.

    Uses Mistral Large model for document processing tasks.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the Bedrock client.

        Args:
            region: AWS region (defaults to settings.aws_region)
            model_id: Bedrock model ID (defaults to settings.bedrock_model_id)
            max_tokens: Maximum tokens for response (defaults to settings.bedrock_max_tokens)
            temperature: Temperature for generation (defaults to settings.bedrock_temperature)
        """
        self.region = region or settings.aws_region
        self.model_id = model_id or settings.bedrock_model_id
        self.max_tokens = max_tokens or settings.bedrock_max_tokens
        self.temperature = temperature or settings.bedrock_temperature

        # Initialize Bedrock runtime client
        self._client = self._create_client()

    def _create_client(self):
        """Create the Bedrock runtime client."""
        client_kwargs = {"region_name": self.region}

        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        return boto3.client("bedrock-runtime", **client_kwargs)

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            multiplier=settings.retry_delay, max=settings.retry_delay * 10
        ),
        retry=retry_if_exception_type((BedrockRateLimitError, BedrockTimeoutError)),
        reraise=True,
    )
    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Invoke the Bedrock model with the given prompt.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            The model's response text

        Raises:
            BedrockClientError: If the invocation fails
        """
        try:
            # Build request body for Mistral
            messages = [{"role": "user", "content": prompt}]

            request_body = {
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }

            if system_prompt:
                request_body["system"] = system_prompt

            logger.debug(f"Invoking Bedrock model: {self.model_id}")

            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Extract text from Mistral response format
            if "outputs" in response_body:
                return response_body["outputs"][0]["text"]
            elif "choices" in response_body:
                return response_body["choices"][0]["message"]["content"]
            elif "content" in response_body:
                if isinstance(response_body["content"], list):
                    return response_body["content"][0]["text"]
                return response_body["content"]
            else:
                logger.warning(f"Unexpected response format: {response_body}")
                return str(response_body)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                logger.warning(f"Rate limit hit: {error_message}")
                raise BedrockRateLimitError(f"Rate limit exceeded: {error_message}")
            elif error_code in ["RequestTimeoutException", "ServiceUnavailable"]:
                logger.warning(f"Timeout error: {error_message}")
                raise BedrockTimeoutError(f"Request timeout: {error_message}")
            else:
                logger.error(f"Bedrock error ({error_code}): {error_message}")
                raise BedrockClientError(f"Bedrock error: {error_message}")

        except BotoCoreError as e:
            logger.error(f"BotoCore error: {e}")
            raise BedrockClientError(f"AWS client error: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Bedrock response: {e}")
            raise BedrockClientError(f"Invalid response format: {e}")

    def invoke_with_json_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the model and parse the response as JSON.

        Args:
            prompt: The prompt requesting JSON output
            system_prompt: Optional system prompt
            temperature: Override default temperature

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            BedrockClientError: If invocation or JSON parsing fails
        """
        response = self.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or 0.0,  # Lower temperature for structured output
        )

        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()

            return json.loads(response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response[:500]}")
            raise BedrockClientError(f"Invalid JSON in response: {e}")

    def health_check(self) -> bool:
        """
        Check if the Bedrock client is healthy and can connect.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.invoke("Say 'healthy' in one word.", max_tokens=10)
            return "healthy" in response.lower()
        except BedrockClientError:
            return False


# Global client instance (lazy initialization)
_bedrock_client: Optional[BedrockClient] = None


def get_bedrock_client() -> BedrockClient:
    """Get or create the global Bedrock client instance."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = BedrockClient()
    return _bedrock_client
