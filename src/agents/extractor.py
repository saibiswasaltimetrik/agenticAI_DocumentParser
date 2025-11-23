"""
Extractor Agent for extracting key fields from documents.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..core.bedrock_client import BedrockClient
from ..schemas.document import (
    AgentType,
    DocumentType,
    ExtractedData,
    ExtractedField,
    ProcessingState,
    ProcessingStatus,
    DOCUMENT_SCHEMAS,
)
from ..utils.logging import ResponsibleAILogger
from .base import BaseAgent

logger = logging.getLogger(__name__)


EXTRACTOR_SYSTEM_PROMPT = """You are a document data extraction expert. Your task is to extract structured information from documents based on the document type.

For each field you extract:
1. Identify the field name
2. Extract the value accurately
3. Provide a confidence score (0.0 to 1.0)
4. Normalize values where appropriate (dates to YYYY-MM-DD, currencies to numeric)

Always respond with valid JSON containing the extracted fields.
"""

EXTRACTION_PROMPTS = {
    DocumentType.INVOICE: """Extract the following fields from this invoice:
- invoice_number: The invoice identifier
- date: Invoice date (normalize to YYYY-MM-DD)
- due_date: Payment due date (normalize to YYYY-MM-DD)
- vendor_name: Name of the vendor/seller
- total_amount: Total amount (numeric only)
- tax_amount: Tax amount if present (numeric only)
- line_items: List of items with description and amount
- billing_address: Billing address if present
- payment_terms: Payment terms if specified

Document Content:
{content}

Respond with JSON:
{{
    "fields": [
        {{"name": "<field_name>", "value": "<value>", "confidence": <0.0-1.0>, "data_type": "<type>"}}
    ],
    "structured_data": {{<key-value pairs>}}
}}
""",
    DocumentType.RECEIPT: """Extract the following fields from this receipt:
- date: Transaction date (normalize to YYYY-MM-DD)
- merchant_name: Name of the merchant/store
- total_amount: Total amount paid (numeric only)
- subtotal: Subtotal before tax (numeric only)
- tax_amount: Tax amount (numeric only)
- payment_method: How payment was made
- items: List of purchased items with prices

Document Content:
{content}

Respond with JSON:
{{
    "fields": [
        {{"name": "<field_name>", "value": "<value>", "confidence": <0.0-1.0>, "data_type": "<type>"}}
    ],
    "structured_data": {{<key-value pairs>}}
}}
""",
    DocumentType.CONTRACT: """Extract the following fields from this contract:
- parties: Names of all parties involved
- effective_date: Contract start date (normalize to YYYY-MM-DD)
- termination_date: Contract end date if specified (normalize to YYYY-MM-DD)
- terms: Key terms and conditions
- governing_law: Applicable law/jurisdiction
- signatures: Signatories if mentioned

Document Content:
{content}

Respond with JSON:
{{
    "fields": [
        {{"name": "<field_name>", "value": "<value>", "confidence": <0.0-1.0>, "data_type": "<type>"}}
    ],
    "structured_data": {{<key-value pairs>}}
}}
""",
    DocumentType.RESUME: """Extract the following fields from this resume/CV:
- name: Full name of the candidate
- contact_info: Email, phone, address
- summary: Professional summary if present
- education: Educational background (list)
- experience: Work experience (list with company, role, dates)
- skills: Listed skills
- certifications: Any certifications

Document Content:
{content}

Respond with JSON:
{{
    "fields": [
        {{"name": "<field_name>", "value": "<value>", "confidence": <0.0-1.0>, "data_type": "<type>"}}
    ],
    "structured_data": {{<key-value pairs>}}
}}
""",
    DocumentType.RESEARCH_PAPER: """Extract the following fields from this research paper:
- title: Paper title
- authors: List of authors
- abstract: Paper abstract
- keywords: Keywords if listed
- institution: Affiliated institutions
- publication_date: Publication date if available
- doi: DOI if present
- references_count: Number of references

Document Content:
{content}

Respond with JSON:
{{
    "fields": [
        {{"name": "<field_name>", "value": "<value>", "confidence": <0.0-1.0>, "data_type": "<type>"}}
    ],
    "structured_data": {{<key-value pairs>}}
}}
""",
}

DEFAULT_EXTRACTION_PROMPT = """Extract all relevant structured information from this document.
Identify key fields such as:
- Names (people, organizations)
- Dates
- Amounts/numbers
- Addresses
- Identifiers (IDs, reference numbers)
- Any other important data

Document Content:
{content}

Respond with JSON:
{{
    "fields": [
        {{"name": "<field_name>", "value": "<value>", "confidence": <0.0-1.0>, "data_type": "<type>"}}
    ],
    "structured_data": {{<key-value pairs>}}
}}
"""


class ExtractorAgent(BaseAgent):
    """
    Agent responsible for extracting structured data from documents.

    Uses document type-specific prompts to extract relevant fields.
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        rai_logger: Optional[ResponsibleAILogger] = None,
    ):
        """Initialize the extractor agent."""
        super().__init__(
            agent_type=AgentType.EXTRACTOR,
            name="ExtractorAgent",
            bedrock_client=bedrock_client,
            rai_logger=rai_logger,
        )

    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Extract structured data from the document.

        Args:
            state: Current processing state with document_type and raw_content

        Returns:
            Updated state with extracted_data
        """
        start_time = time.time()
        state.current_agent = AgentType.EXTRACTOR

        self.logger.info(f"Extracting data from {state.document_type.value if state.document_type else 'unknown'} document")

        try:
            # Get document-specific prompt or use default
            doc_type = state.document_type or DocumentType.UNKNOWN
            prompt_template = EXTRACTION_PROMPTS.get(doc_type, DEFAULT_EXTRACTION_PROMPT)

            # Truncate content if too long
            content = state.raw_content or ""
            if len(content) > 15000:
                content = content[:15000] + "\n...[truncated]..."

            prompt = prompt_template.format(content=content)

            result = self._invoke_llm(
                prompt=prompt,
                system_prompt=EXTRACTOR_SYSTEM_PROMPT,
                temperature=0.1,
                expect_json=True,
            )

            # Parse extraction result
            fields = []
            raw_fields = result.get("fields", [])

            for field_data in raw_fields:
                if isinstance(field_data, dict):
                    field = ExtractedField(
                        name=field_data.get("name", "unknown"),
                        value=field_data.get("value", ""),
                        confidence=float(field_data.get("confidence", 0.5)),
                        data_type=field_data.get("data_type", "string"),
                    )
                    fields.append(field)

            structured_data = result.get("structured_data", {})
            if not structured_data and fields:
                structured_data = {f.name: f.value for f in fields}

            state.extracted_data = ExtractedData(
                document_type=doc_type,
                fields=fields,
                raw_text=state.raw_content,
                structured_data=structured_data,
            )
            state.extraction_complete = True

            duration_ms = (time.time() - start_time) * 1000

            # Calculate average confidence
            avg_confidence = (
                sum(f.confidence for f in fields) / len(fields) if fields else 0.0
            )

            # Log decision
            decision = self._create_decision(
                input_summary=f"Document type: {doc_type.value}, Content: {len(content)} chars",
                output_summary=f"Extracted {len(fields)} fields with avg confidence {avg_confidence:.2f}",
                duration_ms=duration_ms,
                confidence=avg_confidence,
                reasoning=f"Extracted fields: {[f.name for f in fields[:10]]}",
            )
            state.agent_decisions.append(decision)

            if self.rai_logger:
                self._log_decision(
                    input_summary=f"Document type: {doc_type.value}",
                    output_summary=f"Extracted {len(fields)} fields",
                    duration_ms=duration_ms,
                    confidence=avg_confidence,
                )

            self.logger.info(f"Extraction complete: {len(fields)} fields extracted")

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Extraction failed: {str(e)}"
            self.logger.error(error_msg)

            state.extracted_data = ExtractedData(
                document_type=state.document_type or DocumentType.UNKNOWN,
                fields=[],
                structured_data={},
            )
            state.extraction_complete = True
            state.errors.append(error_msg)

            decision = self._create_decision(
                input_summary=f"Document type: {state.document_type.value if state.document_type else 'unknown'}",
                output_summary="Extraction failed",
                duration_ms=duration_ms,
                error_message=error_msg,
            )
            state.agent_decisions.append(decision)

        return state
