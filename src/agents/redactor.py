"""
Redactor Agent for PII detection and masking.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from ..core.bedrock_client import BedrockClient
from ..schemas.document import (
    AgentType,
    PIIEntity,
    PIIType,
    ProcessingState,
    RedactionResult,
)
from ..utils.logging import ResponsibleAILogger
from .base import BaseAgent

logger = logging.getLogger(__name__)


# Regex patterns for common PII types
PII_PATTERNS = {
    PIIType.SSN: [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{3}\s\d{2}\s\d{4}\b",
        r"\b\d{9}\b(?=.*(?:ssn|social|security))",
    ],
    PIIType.EMAIL: [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    ],
    PIIType.PHONE: [
        r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b\d{3}[-.\s]\d{4}\b",
    ],
    PIIType.CREDIT_CARD: [
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        r"\b\d{16}\b",
    ],
    PIIType.IP_ADDRESS: [
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    ],
    PIIType.DATE_OF_BIRTH: [
        r"\b(?:DOB|Date of Birth|Born|Birthday)[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
        r"\b(?:DOB|Date of Birth|Born)[:\s]*(\w+\s+\d{1,2},?\s+\d{4})\b",
    ],
    PIIType.PASSPORT: [
        r"\b[A-Z]{1,2}\d{6,9}\b",
    ],
    PIIType.DRIVERS_LICENSE: [
        r"\b[A-Z]{1,2}\d{6,8}\b",
    ],
}

# Masking patterns for each PII type
PII_MASKS = {
    PIIType.SSN: "***-**-****",
    PIIType.EMAIL: "[EMAIL REDACTED]",
    PIIType.PHONE: "[PHONE REDACTED]",
    PIIType.CREDIT_CARD: "****-****-****-****",
    PIIType.IP_ADDRESS: "[IP REDACTED]",
    PIIType.DATE_OF_BIRTH: "[DOB REDACTED]",
    PIIType.PASSPORT: "[PASSPORT REDACTED]",
    PIIType.DRIVERS_LICENSE: "[DL REDACTED]",
    PIIType.NAME: "[NAME REDACTED]",
    PIIType.ADDRESS: "[ADDRESS REDACTED]",
    PIIType.BANK_ACCOUNT: "[BANK ACCOUNT REDACTED]",
    PIIType.MEDICAL_ID: "[MEDICAL ID REDACTED]",
    PIIType.TAX_ID: "[TAX ID REDACTED]",
}


PII_DETECTION_SYSTEM_PROMPT = """You are a PII (Personally Identifiable Information) detection expert. Your task is to identify all PII in the given text.

Types of PII to detect:
- Names (full names, first names, last names)
- Addresses (street addresses, cities, zip codes)
- Social Security Numbers
- Phone numbers
- Email addresses
- Credit card numbers
- Bank account numbers
- Dates of birth
- Passport numbers
- Driver's license numbers
- Medical IDs
- Tax IDs

Respond with a JSON array of detected PII:
{
    "pii_entities": [
        {
            "type": "<pii_type>",
            "value": "<original_value>",
            "start": <start_index>,
            "end": <end_index>,
            "confidence": <0.0-1.0>
        }
    ]
}

Be thorough but avoid false positives. Only flag information that is clearly PII.
"""

PII_DETECTION_PROMPT = """Analyze the following text and identify all PII (Personally Identifiable Information).

Text:
{text}

Return a JSON object with all PII entities found. Include the exact text, position, type, and confidence.
"""


class RedactorAgent(BaseAgent):
    """
    Agent responsible for detecting and redacting PII from documents.

    Uses both regex patterns and LLM for comprehensive PII detection.
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        rai_logger: Optional[ResponsibleAILogger] = None,
        use_llm_detection: bool = True,
    ):
        """
        Initialize the redactor agent.

        Args:
            bedrock_client: Optional Bedrock client
            rai_logger: Optional Responsible AI logger
            use_llm_detection: Whether to use LLM for additional PII detection
        """
        super().__init__(
            agent_type=AgentType.REDACTOR,
            name="RedactorAgent",
            bedrock_client=bedrock_client,
            rai_logger=rai_logger,
        )
        self.use_llm_detection = use_llm_detection

    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Detect and redact PII from the document.

        Args:
            state: Current processing state with raw_content

        Returns:
            Updated state with redaction_result
        """
        start_time = time.time()
        state.current_agent = AgentType.REDACTOR

        self.logger.info("Starting PII detection and redaction")

        try:
            text = state.raw_content or ""

            # Detect PII using regex patterns
            regex_entities = self._detect_pii_regex(text)

            # Optionally detect PII using LLM
            llm_entities = []
            if self.use_llm_detection and len(text) > 0:
                llm_entities = self._detect_pii_llm(text)

            # Merge and deduplicate entities
            all_entities = self._merge_entities(regex_entities, llm_entities)

            # Perform redaction
            redacted_text, redacted_entities = self._redact_text(text, all_entities)

            # Calculate PII statistics
            pii_by_type = {}
            for entity in redacted_entities:
                pii_type = entity.pii_type.value
                pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1

            state.redaction_result = RedactionResult(
                original_text=text,
                redacted_text=redacted_text,
                pii_entities=redacted_entities,
                total_pii_found=len(redacted_entities),
                pii_by_type=pii_by_type,
            )
            state.redaction_complete = True

            duration_ms = (time.time() - start_time) * 1000

            # Log decision with PII metrics
            decision = self._create_decision(
                input_summary=f"Text with {len(text)} characters",
                output_summary=f"Found {len(redacted_entities)} PII entities: {pii_by_type}",
                duration_ms=duration_ms,
                confidence=0.95 if redacted_entities else 1.0,
                reasoning=f"Regex found: {len(regex_entities)}, LLM found: {len(llm_entities)}",
            )
            state.agent_decisions.append(decision)

            # Log PII metrics for responsible AI
            if self.rai_logger:
                self._log_decision(
                    input_summary=f"Document with {len(text)} chars",
                    output_summary=f"Redacted {len(redacted_entities)} PII entities",
                    duration_ms=duration_ms,
                    confidence=0.95,
                )

                # Log PII-specific metrics (assume all detections are true positives for now)
                self.rai_logger.log_pii_metrics(
                    total_detected=len(redacted_entities),
                    true_positives=len(redacted_entities),
                    false_positives=0,
                    false_negatives=0,
                )

            self.logger.info(
                f"Redaction complete: {len(redacted_entities)} PII entities redacted"
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Redaction failed: {str(e)}"
            self.logger.error(error_msg)

            # Return original text on failure
            state.redaction_result = RedactionResult(
                original_text=state.raw_content or "",
                redacted_text=state.raw_content or "",
                pii_entities=[],
                total_pii_found=0,
            )
            state.redaction_complete = True
            state.errors.append(error_msg)

            decision = self._create_decision(
                input_summary="PII detection attempt",
                output_summary="Redaction failed",
                duration_ms=duration_ms,
                error_message=error_msg,
            )
            state.agent_decisions.append(decision)

        return state

    def _detect_pii_regex(self, text: str) -> List[PIIEntity]:
        """
        Detect PII using regex patterns.

        Args:
            text: Text to analyze

        Returns:
            List of detected PII entities
        """
        entities = []

        for pii_type, patterns in PII_PATTERNS.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        value = match.group(0)
                        # For patterns with groups, get the captured group
                        if match.lastindex:
                            value = match.group(1)

                        entity = PIIEntity(
                            pii_type=pii_type,
                            original_value=value,
                            masked_value=PII_MASKS.get(pii_type, "[REDACTED]"),
                            start_index=match.start(),
                            end_index=match.end(),
                            confidence=0.9,
                            context=text[max(0, match.start() - 20) : match.end() + 20],
                        )
                        entities.append(entity)
                except re.error as e:
                    self.logger.warning(f"Regex error for pattern {pattern}: {e}")

        return entities

    def _detect_pii_llm(self, text: str) -> List[PIIEntity]:
        """
        Detect PII using LLM.

        Args:
            text: Text to analyze

        Returns:
            List of detected PII entities
        """
        try:
            # Truncate text if too long
            analysis_text = text[:8000] if len(text) > 8000 else text

            prompt = PII_DETECTION_PROMPT.format(text=analysis_text)

            result = self._invoke_llm(
                prompt=prompt,
                system_prompt=PII_DETECTION_SYSTEM_PROMPT,
                temperature=0.1,
                expect_json=True,
            )

            entities = []
            raw_entities = result.get("pii_entities", [])

            for entity_data in raw_entities:
                if isinstance(entity_data, dict):
                    pii_type_str = entity_data.get("type", "").lower()

                    # Map LLM type to PIIType enum
                    pii_type = self._map_pii_type(pii_type_str)
                    if pii_type is None:
                        continue

                    value = entity_data.get("value", "")
                    start = entity_data.get("start", 0)
                    end = entity_data.get("end", start + len(value))

                    entity = PIIEntity(
                        pii_type=pii_type,
                        original_value=value,
                        masked_value=PII_MASKS.get(pii_type, "[REDACTED]"),
                        start_index=start,
                        end_index=end,
                        confidence=float(entity_data.get("confidence", 0.8)),
                    )
                    entities.append(entity)

            return entities

        except Exception as e:
            self.logger.warning(f"LLM PII detection failed: {e}")
            return []

    def _map_pii_type(self, type_str: str) -> Optional[PIIType]:
        """Map string PII type to PIIType enum."""
        mapping = {
            "ssn": PIIType.SSN,
            "social_security": PIIType.SSN,
            "social security number": PIIType.SSN,
            "email": PIIType.EMAIL,
            "email_address": PIIType.EMAIL,
            "phone": PIIType.PHONE,
            "phone_number": PIIType.PHONE,
            "telephone": PIIType.PHONE,
            "address": PIIType.ADDRESS,
            "street_address": PIIType.ADDRESS,
            "name": PIIType.NAME,
            "full_name": PIIType.NAME,
            "person_name": PIIType.NAME,
            "date_of_birth": PIIType.DATE_OF_BIRTH,
            "dob": PIIType.DATE_OF_BIRTH,
            "birthday": PIIType.DATE_OF_BIRTH,
            "credit_card": PIIType.CREDIT_CARD,
            "credit card number": PIIType.CREDIT_CARD,
            "bank_account": PIIType.BANK_ACCOUNT,
            "bank account number": PIIType.BANK_ACCOUNT,
            "passport": PIIType.PASSPORT,
            "passport_number": PIIType.PASSPORT,
            "drivers_license": PIIType.DRIVERS_LICENSE,
            "driver's license": PIIType.DRIVERS_LICENSE,
            "ip_address": PIIType.IP_ADDRESS,
            "ip address": PIIType.IP_ADDRESS,
            "medical_id": PIIType.MEDICAL_ID,
            "medical id": PIIType.MEDICAL_ID,
            "tax_id": PIIType.TAX_ID,
            "tax id": PIIType.TAX_ID,
        }
        return mapping.get(type_str.lower().replace(" ", "_"))

    def _merge_entities(
        self, regex_entities: List[PIIEntity], llm_entities: List[PIIEntity]
    ) -> List[PIIEntity]:
        """
        Merge and deduplicate PII entities from regex and LLM detection.

        Args:
            regex_entities: Entities from regex detection
            llm_entities: Entities from LLM detection

        Returns:
            Merged list of unique entities
        """
        all_entities = regex_entities.copy()

        for llm_entity in llm_entities:
            # Check for overlap with existing entities
            is_duplicate = False
            for existing in all_entities:
                if self._entities_overlap(existing, llm_entity):
                    is_duplicate = True
                    # Keep entity with higher confidence
                    if llm_entity.confidence > existing.confidence:
                        all_entities.remove(existing)
                        all_entities.append(llm_entity)
                    break

            if not is_duplicate:
                all_entities.append(llm_entity)

        # Sort by start index
        all_entities.sort(key=lambda e: e.start_index)
        return all_entities

    def _entities_overlap(self, e1: PIIEntity, e2: PIIEntity) -> bool:
        """Check if two entities overlap in position."""
        return not (e1.end_index <= e2.start_index or e2.end_index <= e1.start_index)

    def _redact_text(
        self, text: str, entities: List[PIIEntity]
    ) -> Tuple[str, List[PIIEntity]]:
        """
        Redact PII from text.

        Args:
            text: Original text
            entities: PII entities to redact

        Returns:
            Tuple of (redacted_text, updated_entities)
        """
        if not entities:
            return text, []

        # Sort entities by start index in reverse order for replacement
        sorted_entities = sorted(entities, key=lambda e: e.start_index, reverse=True)

        redacted_text = text
        redacted_entities = []

        for entity in sorted_entities:
            # Find the actual value in text (positions might be off)
            actual_start = redacted_text.find(entity.original_value)
            if actual_start >= 0:
                actual_end = actual_start + len(entity.original_value)
                redacted_text = (
                    redacted_text[:actual_start]
                    + entity.masked_value
                    + redacted_text[actual_end:]
                )
                redacted_entities.append(entity)
            elif entity.start_index < len(redacted_text):
                # Use original indices
                redacted_text = (
                    redacted_text[: entity.start_index]
                    + entity.masked_value
                    + redacted_text[entity.end_index :]
                )
                redacted_entities.append(entity)

        return redacted_text, list(reversed(redacted_entities))
