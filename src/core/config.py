"""
Configuration settings for the Agentic Document Processor.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # AWS Configuration
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        default=None, alias="AWS_SECRET_ACCESS_KEY"
    )

    # Bedrock Configuration
    bedrock_model_id: str = Field(
        default="mistral.mistral-large-2402-v1:0", alias="BEDROCK_MODEL_ID"
    )
    bedrock_max_tokens: int = Field(default=4096, alias="BEDROCK_MAX_TOKENS")
    bedrock_temperature: float = Field(default=0.1, alias="BEDROCK_TEMPERATURE")

    # Retry Configuration
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, alias="RETRY_DELAY")
    retry_backoff: float = Field(default=2.0, alias="RETRY_BACKOFF")
    request_timeout: int = Field(default=60, alias="REQUEST_TIMEOUT")

    # Processing Configuration
    max_document_size_mb: int = Field(default=10, alias="MAX_DOCUMENT_SIZE_MB")
    ocr_enabled: bool = Field(default=True, alias="OCR_ENABLED")
    ocr_language: str = Field(default="eng", alias="OCR_LANGUAGE")

    # Validation Thresholds
    extraction_accuracy_threshold: float = Field(
        default=0.90, alias="EXTRACTION_ACCURACY_THRESHOLD"
    )
    pii_recall_threshold: float = Field(default=0.95, alias="PII_RECALL_THRESHOLD")
    pii_precision_threshold: float = Field(
        default=0.90, alias="PII_PRECISION_THRESHOLD"
    )
    workflow_success_threshold: float = Field(
        default=0.90, alias="WORKFLOW_SUCCESS_THRESHOLD"
    )
    p95_latency_threshold: float = Field(
        default=4.0, alias="P95_LATENCY_THRESHOLD"
    )  # seconds

    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    responsible_ai_log_file: str = Field(
        default="responsible_ai_log.json", alias="RESPONSIBLE_AI_LOG_FILE"
    )

    # Output Configuration
    output_dir: str = Field(default="output", alias="OUTPUT_DIR")
    metrics_file: str = Field(default="metrics_report.json", alias="METRICS_FILE")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings
