"""
Tests for document processing agents.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.schemas.document import (
    DocumentType,
    DocumentMetadata,
    ExtractedData,
    ExtractedField,
    ProcessingState,
    ProcessingStatus,
    PIIType,
)
from src.agents.classifier import ClassifierAgent
from src.agents.extractor import ExtractorAgent
from src.agents.validator import ValidatorAgent
from src.agents.redactor import RedactorAgent
from src.agents.reporter import ReporterAgent


class TestClassifierAgent:
    """Tests for ClassifierAgent."""

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classifier_initialization(self, mock_get_client):
        """Test classifier agent initialization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        agent = ClassifierAgent()

        assert agent.name == "ClassifierAgent"
        assert agent.agent_type.value == "classifier"

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classify_invoice(self, mock_get_client, sample_processing_state):
        """Test classifying an invoice document."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = {
            "document_type": "invoice",
            "confidence": 0.95,
            "reasoning": "Contains invoice number and billing information",
        }
        mock_client.model_id = "mistral.mistral-large-2402-v1:0"
        mock_get_client.return_value = mock_client

        agent = ClassifierAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.document_type == DocumentType.INVOICE
        assert result_state.classification_confidence == 0.95
        assert len(result_state.agent_decisions) == 1

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classify_unknown_type(self, mock_get_client, sample_processing_state):
        """Test handling unknown document type."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = {
            "document_type": "something_invalid",
            "confidence": 0.3,
            "reasoning": "Could not determine type",
        }
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        agent = ClassifierAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.document_type == DocumentType.UNKNOWN

    @patch("src.agents.classifier.get_bedrock_client")
    def test_classify_handles_error(self, mock_get_client, sample_processing_state):
        """Test error handling in classification."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.side_effect = Exception("API Error")
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        agent = ClassifierAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.document_type == DocumentType.UNKNOWN
        assert result_state.classification_confidence == 0.0
        assert len(result_state.errors) > 0


class TestExtractorAgent:
    """Tests for ExtractorAgent."""

    @patch("src.agents.extractor.get_bedrock_client")
    def test_extractor_initialization(self, mock_get_client):
        """Test extractor agent initialization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        agent = ExtractorAgent()

        assert agent.name == "ExtractorAgent"
        assert agent.agent_type.value == "extractor"

    @patch("src.agents.extractor.get_bedrock_client")
    def test_extract_invoice_fields(
        self, mock_get_client, sample_processing_state, mock_bedrock_extractor_response
    ):
        """Test extracting fields from an invoice."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = mock_bedrock_extractor_response
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.document_type = DocumentType.INVOICE

        agent = ExtractorAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.extraction_complete is True
        assert result_state.extracted_data is not None
        assert len(result_state.extracted_data.fields) > 0
        assert "invoice_number" in result_state.extracted_data.structured_data

    @patch("src.agents.extractor.get_bedrock_client")
    def test_extract_handles_empty_response(self, mock_get_client, sample_processing_state):
        """Test handling empty extraction response."""
        mock_client = MagicMock()
        mock_client.invoke_with_json_output.return_value = {
            "fields": [],
            "structured_data": {},
        }
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.document_type = DocumentType.INVOICE

        agent = ExtractorAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.extraction_complete is True
        assert len(result_state.extracted_data.fields) == 0


class TestValidatorAgent:
    """Tests for ValidatorAgent."""

    @patch("src.agents.validator.get_bedrock_client")
    def test_validator_initialization(self, mock_get_client):
        """Test validator agent initialization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        agent = ValidatorAgent()

        assert agent.name == "ValidatorAgent"
        assert agent.agent_type.value == "validator"

    @patch("src.agents.validator.get_bedrock_client")
    def test_validate_valid_invoice(self, mock_get_client, sample_processing_state, sample_extracted_data):
        """Test validating a valid invoice."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.document_type = DocumentType.INVOICE
        sample_processing_state.extracted_data = sample_extracted_data

        agent = ValidatorAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.validation_complete is True
        assert result_state.validation_result is not None
        assert result_state.validation_result.is_valid is True

    @patch("src.agents.validator.get_bedrock_client")
    def test_validate_missing_required_fields(self, mock_get_client, sample_processing_state):
        """Test validation fails with missing required fields."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.document_type = DocumentType.INVOICE
        sample_processing_state.extracted_data = ExtractedData(
            document_type=DocumentType.INVOICE,
            fields=[],
            structured_data={},  # Missing all required fields
        )

        agent = ValidatorAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.validation_complete is True
        assert result_state.validation_result.is_valid is False
        assert len(result_state.validation_result.errors) > 0

    @patch("src.agents.validator.get_bedrock_client")
    def test_validate_pattern_mismatch(self, mock_get_client, sample_processing_state):
        """Test validation fails with pattern mismatch."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.document_type = DocumentType.INVOICE
        sample_processing_state.extracted_data = ExtractedData(
            document_type=DocumentType.INVOICE,
            fields=[],
            structured_data={
                "invoice_number": "INV-2024-001234",
                "date": "November 15, 2024",  # Wrong format
                "total_amount": "$2,495.50",  # Contains symbols
                "vendor_name": "ABC Company",
            },
        )

        agent = ValidatorAgent(bedrock_client=mock_client)
        result_state = agent.process(sample_processing_state)

        assert result_state.validation_complete is True
        # Should have errors for date format
        error_fields = [e.field_name for e in result_state.validation_result.errors]
        assert "date" in error_fields


class TestRedactorAgent:
    """Tests for RedactorAgent."""

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redactor_initialization(self, mock_get_client):
        """Test redactor agent initialization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        agent = RedactorAgent(use_llm_detection=False)

        assert agent.name == "RedactorAgent"
        assert agent.agent_type.value == "redactor"

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_email(self, mock_get_client, sample_processing_state):
        """Test redacting email addresses."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.raw_content = "Contact: john.doe@email.com for support."

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(sample_processing_state)

        assert result_state.redaction_complete is True
        assert result_state.redaction_result is not None
        assert "[EMAIL REDACTED]" in result_state.redaction_result.redacted_text
        assert result_state.redaction_result.total_pii_found >= 1

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_phone(self, mock_get_client, sample_processing_state):
        """Test redacting phone numbers."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.raw_content = "Call us at (555) 123-4567 or 555-987-6543."

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(sample_processing_state)

        assert result_state.redaction_complete is True
        assert "[PHONE REDACTED]" in result_state.redaction_result.redacted_text

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_ssn(self, mock_get_client, sample_processing_state):
        """Test redacting SSN."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.raw_content = "SSN: 123-45-6789"

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(sample_processing_state)

        assert result_state.redaction_complete is True
        assert "***-**-****" in result_state.redaction_result.redacted_text

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_credit_card(self, mock_get_client, sample_processing_state):
        """Test redacting credit card numbers."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.raw_content = "Card: 4532-1234-5678-9012"

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(sample_processing_state)

        assert result_state.redaction_complete is True
        assert "****-****-****-****" in result_state.redaction_result.redacted_text

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_no_pii(self, mock_get_client, sample_processing_state):
        """Test handling content with no PII."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.raw_content = "This is a clean document with no personal information."

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(sample_processing_state)

        assert result_state.redaction_complete is True
        assert result_state.redaction_result.total_pii_found == 0
        assert result_state.redaction_result.redacted_text == sample_processing_state.raw_content

    @patch("src.agents.redactor.get_bedrock_client")
    def test_redact_multiple_pii(self, mock_get_client, sample_processing_state, sample_invoice_content):
        """Test redacting multiple PII types."""
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_get_client.return_value = mock_client

        sample_processing_state.raw_content = sample_invoice_content

        agent = RedactorAgent(bedrock_client=mock_client, use_llm_detection=False)
        result_state = agent.process(sample_processing_state)

        assert result_state.redaction_complete is True
        assert result_state.redaction_result.total_pii_found > 0
        # Should find email, phone, SSN, credit card
        pii_types_found = result_state.redaction_result.pii_by_type
        assert len(pii_types_found) >= 2


class TestReporterAgent:
    """Tests for ReporterAgent."""

    def test_reporter_initialization(self, temp_output_dir):
        """Test reporter agent initialization."""
        agent = ReporterAgent(output_dir=str(temp_output_dir))

        assert agent.name == "ReporterAgent"
        assert agent.agent_type.value == "reporter"

    def test_generate_result(self, sample_processing_state, sample_extracted_data, temp_output_dir):
        """Test generating processing result."""
        sample_processing_state.document_type = DocumentType.INVOICE
        sample_processing_state.extracted_data = sample_extracted_data
        sample_processing_state.extraction_complete = True
        sample_processing_state.validation_complete = True
        sample_processing_state.redaction_complete = True
        sample_processing_state.start_time = datetime.utcnow()
        sample_processing_state.end_time = datetime.utcnow()

        agent = ReporterAgent(output_dir=str(temp_output_dir))
        result = agent.generate_result(sample_processing_state)

        assert result is not None
        assert result.document_type == DocumentType.INVOICE
        assert "invoice_number" in result.extracted_data

    def test_save_result(self, sample_processing_state, sample_extracted_data, temp_output_dir):
        """Test saving result to file."""
        sample_processing_state.document_type = DocumentType.INVOICE
        sample_processing_state.extracted_data = sample_extracted_data
        sample_processing_state.start_time = datetime.utcnow()
        sample_processing_state.end_time = datetime.utcnow()

        agent = ReporterAgent(output_dir=str(temp_output_dir))
        result = agent.generate_result(sample_processing_state)
        output_path = agent.save_result(result, "test_result.json")

        assert output_path is not None
        from pathlib import Path
        assert Path(output_path).exists()
