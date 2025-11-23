"""
Classifier Agent for document type classification.
"""

import logging
import time
from typing import Optional

from ..core.bedrock_client import BedrockClient
from ..schemas.document import (
    AgentType,
    DocumentType,
    ProcessingState,
    ProcessingStatus,
)
from ..utils.logging import ResponsibleAILogger
from .base import BaseAgent

logger = logging.getLogger(__name__)


CLASSIFIER_SYSTEM_PROMPT = """You are a document classification expert. Your task is to analyze document content and classify it into one of the following categories:

- invoice: Bills, payment requests, commercial invoices
- receipt: Purchase receipts, payment confirmations
- contract: Legal agreements, terms of service, contracts
- report: Business reports, analysis documents
- letter: Formal correspondence, letters
- form: Application forms, questionnaires
- resume: CVs, job applications, professional profiles
- id_document: Identity documents, passports, licenses
- medical_record: Health records, prescriptions, medical reports
- financial_statement: Balance sheets, income statements, financial reports
- research_paper: Academic papers, scientific articles
- unknown: Cannot be classified into above categories

Respond with a JSON object containing:
- document_type: The classified type (one of the above)
- confidence: Your confidence score (0.0 to 1.0)
- reasoning: Brief explanation for your classification
"""

CLASSIFIER_PROMPT_TEMPLATE = """Analyze the following document content and classify it.

Document Content:
{content}

Provide your classification as JSON:
{{
    "document_type": "<type>",
    "confidence": <0.0-1.0>,
    "reasoning": "<explanation>"
}}
"""


class ClassifierAgent(BaseAgent):
    """
    Agent responsible for classifying document types.

    Uses LLM to analyze document content and determine the document type
    with a confidence score.
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        rai_logger: Optional[ResponsibleAILogger] = None,
    ):
        """Initialize the classifier agent."""
        super().__init__(
            agent_type=AgentType.CLASSIFIER,
            name="ClassifierAgent",
            bedrock_client=bedrock_client,
            rai_logger=rai_logger,
        )

    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Classify the document type.

        Args:
            state: Current processing state with raw_content

        Returns:
            Updated state with document_type and classification_confidence
        """
        start_time = time.time()
        state.current_agent = AgentType.CLASSIFIER
        state.status = ProcessingStatus.IN_PROGRESS

        self.logger.info(f"Classifying document: {state.document_metadata.file_name if state.document_metadata else 'unknown'}")

        try:
            # Truncate content if too long
            content = state.raw_content or ""
            if len(content) > 10000:
                content = content[:10000] + "\n...[truncated]..."

            prompt = CLASSIFIER_PROMPT_TEMPLATE.format(content=content)

            result = self._invoke_llm(
                prompt=prompt,
                system_prompt=CLASSIFIER_SYSTEM_PROMPT,
                temperature=0.1,
                expect_json=True,
            )

            # Parse classification result
            doc_type_str = result.get("document_type", "unknown").lower()
            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                doc_type = DocumentType.UNKNOWN
                self.logger.warning(f"Unknown document type returned: {doc_type_str}")

            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")

            state.document_type = doc_type
            state.classification_confidence = confidence

            duration_ms = (time.time() - start_time) * 1000

            # Log decision
            decision = self._create_decision(
                input_summary=f"Document: {state.document_metadata.file_name if state.document_metadata else 'unknown'} ({len(content)} chars)",
                output_summary=f"Classified as {doc_type.value} with {confidence:.2f} confidence",
                duration_ms=duration_ms,
                confidence=confidence,
                reasoning=reasoning,
            )
            state.agent_decisions.append(decision)

            if self.rai_logger:
                self._log_decision(
                    input_summary=f"Document content ({len(content)} chars)",
                    output_summary=f"Type: {doc_type.value}, Confidence: {confidence:.2f}",
                    duration_ms=duration_ms,
                    confidence=confidence,
                    reasoning=reasoning,
                )

            self.logger.info(f"Classification complete: {doc_type.value} ({confidence:.2f})")

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Classification failed: {str(e)}"
            self.logger.error(error_msg)

            state.document_type = DocumentType.UNKNOWN
            state.classification_confidence = 0.0
            state.errors.append(error_msg)

            decision = self._create_decision(
                input_summary=f"Document: {state.document_metadata.file_name if state.document_metadata else 'unknown'}",
                output_summary="Classification failed",
                duration_ms=duration_ms,
                error_message=error_msg,
            )
            state.agent_decisions.append(decision)

        return state
