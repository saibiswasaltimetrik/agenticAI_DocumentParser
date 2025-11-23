"""
Base agent class for the document processing pipeline.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..core.bedrock_client import BedrockClient, BedrockClientError, get_bedrock_client
from ..core.config import settings
from ..schemas.document import AgentDecision, AgentType, ProcessingState
from ..utils.logging import ResponsibleAILogger


class BaseAgent(ABC):
    """
    Base class for all agents in the document processing pipeline.

    Provides common functionality for LLM invocation, logging, and error handling.
    """

    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        bedrock_client: Optional[BedrockClient] = None,
        rai_logger: Optional[ResponsibleAILogger] = None,
    ):
        """
        Initialize the base agent.

        Args:
            agent_type: Type of this agent
            name: Name/identifier for this agent
            bedrock_client: Optional Bedrock client (uses global if not provided)
            rai_logger: Optional Responsible AI logger
        """
        self.agent_type = agent_type
        self.name = name
        self.bedrock_client = bedrock_client or get_bedrock_client()
        self.rai_logger = rai_logger
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Process the current state and return updated state.

        Args:
            state: Current processing state

        Returns:
            Updated processing state
        """
        pass

    def _invoke_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        expect_json: bool = False,
    ) -> Any:
        """
        Invoke the LLM with retry logic.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            expect_json: Whether to parse response as JSON

        Returns:
            LLM response (string or dict if expect_json)
        """
        if expect_json:
            return self.bedrock_client.invoke_with_json_output(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        return self.bedrock_client.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    def _log_decision(
        self,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_count: int = 0,
    ) -> Optional[AgentDecision]:
        """
        Log an agent decision to the RAI logger.

        Args:
            input_summary: Summary of input
            output_summary: Summary of output
            duration_ms: Processing duration
            confidence: Confidence score
            reasoning: Reasoning explanation
            error_message: Error if any
            retry_count: Number of retries

        Returns:
            AgentDecision if logger is available
        """
        if self.rai_logger:
            return self.rai_logger.log_decision(
                agent_type=self.agent_type,
                agent_name=self.name,
                input_summary=input_summary,
                output_summary=output_summary,
                duration_ms=duration_ms,
                confidence=confidence,
                reasoning=reasoning,
                model_used=self.bedrock_client.model_id,
                retry_count=retry_count,
                error_message=error_message,
            )
        return None

    def _create_decision(
        self,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> AgentDecision:
        """Create an AgentDecision object."""
        return AgentDecision(
            agent_type=self.agent_type,
            agent_name=self.name,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            confidence=confidence,
            reasoning=reasoning,
            model_used=self.bedrock_client.model_id,
            error_message=error_message,
        )
