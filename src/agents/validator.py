"""
Validator Agent for JSON-Schema validation and self-repair.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..core.bedrock_client import BedrockClient
from ..schemas.document import (
    AgentType,
    DocumentType,
    ProcessingState,
    ValidationError,
    ValidationResult,
    DOCUMENT_SCHEMAS,
)
from ..utils.logging import ResponsibleAILogger
from .base import BaseAgent

logger = logging.getLogger(__name__)


REPAIR_SYSTEM_PROMPT = """You are a data repair expert. Your task is to fix validation errors in extracted document data.

Given the original extracted data and a list of validation errors, provide corrected values that satisfy the validation requirements.

Respond with JSON containing only the corrected fields:
{
    "corrected_fields": {
        "field_name": "corrected_value"
    },
    "repair_notes": "explanation of changes made"
}
"""

REPAIR_PROMPT_TEMPLATE = """The following extracted data has validation errors that need to be fixed.

Original Data:
{original_data}

Validation Errors:
{errors}

Document Content (for reference):
{content}

Please provide corrected values for the fields with errors. Ensure:
1. Dates are in YYYY-MM-DD format
2. Numbers are numeric only (no currency symbols)
3. Required fields have values
4. Values match expected patterns

Respond with JSON:
{{
    "corrected_fields": {{<field_name>: <corrected_value>}},
    "repair_notes": "<explanation>"
}}
"""


class ValidatorAgent(BaseAgent):
    """
    Agent responsible for validating extracted data against schemas
    and performing self-repair when validation fails.
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        rai_logger: Optional[ResponsibleAILogger] = None,
    ):
        """Initialize the validator agent."""
        super().__init__(
            agent_type=AgentType.VALIDATOR,
            name="ValidatorAgent",
            bedrock_client=bedrock_client,
            rai_logger=rai_logger,
        )

    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Validate extracted data and attempt repair if needed.

        Args:
            state: Current processing state with extracted_data

        Returns:
            Updated state with validation_result
        """
        start_time = time.time()
        state.current_agent = AgentType.VALIDATOR

        self.logger.info("Validating extracted data")

        try:
            # Get schema for document type
            doc_type = state.document_type or DocumentType.UNKNOWN
            schema = DOCUMENT_SCHEMAS.get(doc_type, {})

            # Get extracted data
            if not state.extracted_data:
                state.validation_result = ValidationResult(
                    is_valid=False,
                    errors=[
                        ValidationError(
                            field_name="extracted_data",
                            error_type="missing_data",
                            error_message="No extracted data to validate",
                            severity="error",
                        )
                    ],
                )
                state.validation_complete = True
                return state

            extracted_dict = state.extracted_data.structured_data

            # Perform validation
            errors, warnings = self._validate_against_schema(
                extracted_dict, schema, doc_type
            )

            # If validation fails and repair not exhausted, attempt repair
            if errors and state.repair_attempts < state.max_repair_attempts:
                self.logger.info(
                    f"Validation failed with {len(errors)} errors. Attempting repair (attempt {state.repair_attempts + 1})"
                )
                state.needs_repair = True
                state.repair_attempts += 1

                repaired_data = self._attempt_repair(
                    extracted_dict,
                    errors,
                    state.raw_content or "",
                )

                if repaired_data:
                    # Update extracted data with repairs
                    state.extracted_data.structured_data.update(repaired_data)

                    # Re-validate
                    errors, warnings = self._validate_against_schema(
                        state.extracted_data.structured_data, schema, doc_type
                    )

            is_valid = len(errors) == 0
            state.validation_result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                repair_attempted=state.repair_attempts > 0,
                repair_successful=state.repair_attempts > 0 and is_valid,
            )
            state.validation_complete = True
            state.needs_repair = len(errors) > 0 and state.repair_attempts < state.max_repair_attempts

            duration_ms = (time.time() - start_time) * 1000

            # Log decision
            decision = self._create_decision(
                input_summary=f"Validating {len(extracted_dict)} fields for {doc_type.value}",
                output_summary=f"Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}",
                duration_ms=duration_ms,
                confidence=1.0 if is_valid else 0.5,
                reasoning=f"Repair attempts: {state.repair_attempts}",
            )
            state.agent_decisions.append(decision)

            if self.rai_logger:
                self._log_decision(
                    input_summary=f"Document type: {doc_type.value}",
                    output_summary=f"Valid: {is_valid}",
                    duration_ms=duration_ms,
                    confidence=1.0 if is_valid else 0.5,
                )

            self.logger.info(
                f"Validation complete: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}"
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Validation failed: {str(e)}"
            self.logger.error(error_msg)

            state.validation_result = ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        field_name="validation",
                        error_type="validation_error",
                        error_message=error_msg,
                        severity="error",
                    )
                ],
            )
            state.validation_complete = True
            state.errors.append(error_msg)

            decision = self._create_decision(
                input_summary="Validation attempt",
                output_summary="Validation failed with exception",
                duration_ms=duration_ms,
                error_message=error_msg,
            )
            state.agent_decisions.append(decision)

        return state

    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        doc_type: DocumentType,
    ) -> tuple[List[ValidationError], List[ValidationError]]:
        """
        Validate data against schema.

        Args:
            data: Extracted data dictionary
            schema: Schema with required/optional fields and patterns
            doc_type: Document type

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        required_fields = schema.get("required_fields", [])
        field_patterns = schema.get("field_patterns", {})

        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                errors.append(
                    ValidationError(
                        field_name=field,
                        error_type="missing_required",
                        error_message=f"Required field '{field}' is missing or empty",
                        severity="error",
                    )
                )

        # Check field patterns
        for field, pattern in field_patterns.items():
            if field in data and data[field]:
                value = str(data[field])
                if not re.match(pattern, value):
                    errors.append(
                        ValidationError(
                            field_name=field,
                            error_type="pattern_mismatch",
                            error_message=f"Field '{field}' value '{value}' does not match expected pattern",
                            expected_value=pattern,
                            actual_value=value,
                            severity="error",
                        )
                    )

        # Additional type-specific validations
        if doc_type == DocumentType.INVOICE:
            self._validate_invoice_specific(data, errors, warnings)
        elif doc_type == DocumentType.RECEIPT:
            self._validate_receipt_specific(data, errors, warnings)

        return errors, warnings

    def _validate_invoice_specific(
        self,
        data: Dict[str, Any],
        errors: List[ValidationError],
        warnings: List[ValidationError],
    ):
        """Invoice-specific validation rules."""
        # Check if total_amount is numeric
        if "total_amount" in data:
            try:
                float(str(data["total_amount"]).replace(",", "").replace("$", ""))
            except ValueError:
                errors.append(
                    ValidationError(
                        field_name="total_amount",
                        error_type="invalid_type",
                        error_message="total_amount must be numeric",
                        actual_value=data["total_amount"],
                        severity="error",
                    )
                )

    def _validate_receipt_specific(
        self,
        data: Dict[str, Any],
        errors: List[ValidationError],
        warnings: List[ValidationError],
    ):
        """Receipt-specific validation rules."""
        # Similar numeric validation for amounts
        for field in ["total_amount", "subtotal", "tax_amount"]:
            if field in data and data[field]:
                try:
                    float(str(data[field]).replace(",", "").replace("$", ""))
                except ValueError:
                    errors.append(
                        ValidationError(
                            field_name=field,
                            error_type="invalid_type",
                            error_message=f"{field} must be numeric",
                            actual_value=data[field],
                            severity="error",
                        )
                    )

    def _attempt_repair(
        self,
        data: Dict[str, Any],
        errors: List[ValidationError],
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair validation errors using LLM.

        Args:
            data: Original extracted data
            errors: List of validation errors
            content: Original document content

        Returns:
            Dictionary of corrected fields or None
        """
        try:
            # Format errors for prompt
            error_descriptions = []
            for error in errors:
                error_descriptions.append(
                    f"- {error.field_name}: {error.error_message}"
                    + (f" (expected: {error.expected_value})" if error.expected_value else "")
                )

            prompt = REPAIR_PROMPT_TEMPLATE.format(
                original_data=str(data),
                errors="\n".join(error_descriptions),
                content=content[:5000] if len(content) > 5000 else content,
            )

            result = self._invoke_llm(
                prompt=prompt,
                system_prompt=REPAIR_SYSTEM_PROMPT,
                temperature=0.1,
                expect_json=True,
            )

            corrected = result.get("corrected_fields", {})
            repair_notes = result.get("repair_notes", "")

            self.logger.info(f"Repair attempt: {repair_notes}")
            return corrected

        except Exception as e:
            self.logger.warning(f"Repair attempt failed: {e}")
            return None
