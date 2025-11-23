"""Core modules for configuration and Bedrock client."""

from .config import settings
from .bedrock_client import BedrockClient

__all__ = ["settings", "BedrockClient"]
