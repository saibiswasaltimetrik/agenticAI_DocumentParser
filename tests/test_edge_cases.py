"""
Tests for edge cases: OCR noise, missing fields, timeouts, etc.
"""

import pytest
from unittest.mock import MagicMock, patch
from tenacity import RetryError

from src.schemas.document import (
    DocumentType,
    ExtractedData,
    ProcessingState,
    ProcessingStatus,
)
from src.agents.classifier import ClassifierAgent
from src.agents.extractor import ExtractorAgent
from src.agents.validator import ValidatorAgent
from src.agents.redactor import RedactorAgent
from src.core.bedrock_client import BedrockClientError, BedrockTimeoutError


class TestOCRNoiseHandling:
    """Tests for handling OCR noise in documents."""

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classify_noisy_content(self, mock_get_client, noisy_ocr_content, sample_document_metadata):
        """Test classification with OCR noise."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = {
            "document_type": "invoice",
            "confidence": 0.75,  # Lower confidence due to noise
            "reasoning": "Appears to be an invoice despite OCR errors",
        }
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content=noisy_ocr_content,
        )

        agent = ClassifierAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        assert result_state.document_type == DocumentType.INVOICE
        assert result_state.classification_confidence <= 0.80  # Lower confidence expected

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_noisy_pii(self, mock_get_client, noisy_ocr_content, sample_document_metadata):
        """Test PII redaction with OCR noise."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content=noisy_ocr_content,
        )

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(state)

        # Regex-based detection should still find some patterns
        assert result_state.redaction_complete is True


class TestMissingFieldsHandling:
    """Tests for handling documents with missing required fields."""

    @patch("src.agents.validator.get_bedrock_client")
    def test_validate_missing_required_fields(self, mock_get_client, missing_fields_content, sample_document_metadata):
        """Test validation with missing required fields."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_client.invoke_with_json_output.return_value = {
            "corrected_fields": {},
            "repair_notes": "Could not find required fields in document",
        }
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content=missing_fields_content,
            document_type=DocumentType.INVOICE,
            extracted_data=ExtractedData(
                document_type=DocumentType.INVOICE,
                fields=[],
                structured_data={},
            ),
        )

        agent = ValidatorAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        assert result_state.validation_complete is True
        assert result_state.validation_result.is_valid is False
        assert len(result_state.validation_result.errors) >= 4  # All required fields missing

    @patch("src.agents.extractor.get_bedrock_client")
    def test_extract_from_minimal_content(self, mock_get_client, sample_document_metadata):
        """Test extraction from minimal content."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = {
            "fields": [],
            "structured_data": {},
        }
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="Just some random text.",
            document_type=DocumentType.UNKNOWN,
        )

        agent = ExtractorAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        assert result_state.extraction_complete is True
        assert len(result_state.extracted_data.fields) == 0


class TestBedrockTimeoutHandling:
    """Tests for Bedrock timeout and retry behavior."""

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classifier_handles_timeout(self, mock_get_client, sample_document_metadata):
        """Test classifier handles Bedrock timeout gracefully."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.side_effect = BedrockTimeoutError("Request timed out")
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="Invoice content here",
        )

        agent = ClassifierAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        # Should handle error gracefully
        assert result_state.document_type == DocumentType.UNKNOWN
        assert len(result_state.errors) > 0

    @patch("src.agents.extractor.get_bedrock_client")
    def test_extractor_handles_api_error(self, mock_get_client, sample_document_metadata):
        """Test extractor handles API errors."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.side_effect = BedrockClientError("API Error")
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="Invoice content here",
            document_type=DocumentType.INVOICE,
        )

        agent = ExtractorAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        # Should handle error and still complete
        assert result_state.extraction_complete is True
        assert len(result_state.errors) > 0


class TestLargeDocumentHandling:
    """Tests for handling large documents."""

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classify_large_document(self, mock_get_client, sample_document_metadata):
        """Test classification of large document content."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = {
            "document_type": "report",
            "confidence": 0.85,
            "reasoning": "Large document content",
        }
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        # Create large content (> 10000 chars)
        large_content = "This is a test document. " * 1000

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content=large_content,
        )

        agent = ClassifierAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        # Should handle large content by truncation
        assert result_state.document_type is not None

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_large_document(self, mock_get_client, sample_document_metadata):
        """Test PII redaction in large document."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        # Create large content with PII
        large_content = "Contact: test@email.com. " * 500

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content=large_content,
        )

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(state)

        assert result_state.redaction_complete is True
        # Should find many email addresses
        assert result_state.redaction_result.total_pii_found >= 100


class TestValidationRepair:
    """Tests for self-repair validation mechanism."""

    @patch("src.agents.validator.get_bedrock_client")
    def test_validation_repair_success(self, mock_get_client, sample_document_metadata):
        """Test successful validation repair."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"

        # First repair call should fix the date format
        mock_client.invoke_with_json_output.return_value = {
            "corrected_fields": {
                "date": "2024-11-15",
            },
            "repair_notes": "Fixed date format",
        }
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="Invoice with date November 15, 2024",
            document_type=DocumentType.INVOICE,
            extracted_data=ExtractedData(
                document_type=DocumentType.INVOICE,
                fields=[],
                structured_data={
                    "invoice_number": "INV-001",
                    "date": "November 15, 2024",  # Wrong format
                    "total_amount": "100.00",
                    "vendor_name": "Test",
                },
            ),
        )

        agent = ValidatorAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        assert result_state.validation_complete is True
        assert result_state.repair_attempts >= 1

    @patch("src.agents.validator.get_bedrock_client")
    def test_validation_repair_max_attempts(self, mock_get_client, sample_document_metadata):
        """Test validation stops after max repair attempts."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"

        # Always return invalid repairs
        mock_client.invoke_with_json_output.return_value = {
            "corrected_fields": {"date": "still-wrong"},
            "repair_notes": "Attempted fix",
        }
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="Invoice content",
            document_type=DocumentType.INVOICE,
            extracted_data=ExtractedData(
                document_type=DocumentType.INVOICE,
                fields=[],
                structured_data={
                    "invoice_number": "INV-001",
                    "date": "bad-date",
                    "total_amount": "100.00",
                    "vendor_name": "Test",
                },
            ),
            max_repair_attempts=2,
        )

        agent = ValidatorAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        # Should stop after max attempts
        assert result_state.repair_attempts <= state.max_repair_attempts


class TestEmptyContentHandling:
    """Tests for handling empty or None content."""

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classify_empty_content(self, mock_get_client, sample_document_metadata):
        """Test classification with empty content."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = {
            "document_type": "unknown",
            "confidence": 0.0,
            "reasoning": "No content to analyze",
        }
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="",
        )

        agent = ClassifierAgent(bedrock_client=mock_client)
        result_state = agent.process(state)

        assert result_state.document_type == DocumentType.UNKNOWN

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_empty_content(self, mock_get_client, sample_document_metadata):
        """Test redaction with empty content."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="",
        )

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(state)

        assert result_state.redaction_complete is True
        assert result_state.redaction_result.total_pii_found == 0
