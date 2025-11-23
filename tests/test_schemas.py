"""
Tests for Pydantic schemas.
"""

import pytest
from datetime import datetime

from src.schemas.document import (
    DocumentType,
    DocumentMetadata,
    ExtractedField,
    ExtractedData,
    PIIEntity,
    PIIType,
    RedactionResult,
    ValidationError,
    ValidationResult,
    AgentDecision,
    AgentType,
    ProcessingState,
    ProcessingStatus,
    ProcessingResult,
    MetricsReport,
    DOCUMENT_SCHEMAS,
)


class TestDocumentMetadata:
    """Tests for DocumentMetadata schema."""

    def test_create_metadata(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            file_path="/path/to/document.pdf",
            file_name="document.pdf",
            file_type="pdf",
            file_size_bytes=1024,
        )

        assert metadata.file_name == "document.pdf"
        assert metadata.file_type == "pdf"
        assert metadata.file_size_bytes == 1024
        assert metadata.page_count is None
        assert metadata.ocr_applied is False

    def test_metadata_with_all_fields(self):
        """Test metadata with all optional fields."""
        metadata = DocumentMetadata(
            file_path="/path/to/doc.pdf",
            file_name="doc.pdf",
            file_type="pdf",
            file_size_bytes=2048,
            page_count=5,
            ocr_applied=True,
            source_hash="abc123",
        )

        assert metadata.page_count == 5
        assert metadata.ocr_applied is True
        assert metadata.source_hash == "abc123"


class TestExtractedField:
    """Tests for ExtractedField schema."""

    def test_create_field(self):
        """Test creating an extracted field."""
        field = ExtractedField(
            name="invoice_number",
            value="INV-001",
            confidence=0.95,
        )

        assert field.name == "invoice_number"
        assert field.value == "INV-001"
        assert field.confidence == 0.95

    def test_confidence_bounds(self):
        """Test confidence score bounds."""
        # Valid confidence
        field = ExtractedField(name="test", value="value", confidence=0.5)
        assert field.confidence == 0.5

        # Boundary values
        field_low = ExtractedField(name="test", value="value", confidence=0.0)
        assert field_low.confidence == 0.0

        field_high = ExtractedField(name="test", value="value", confidence=1.0)
        assert field_high.confidence == 1.0


class TestPIIEntity:
    """Tests for PIIEntity schema."""

    def test_create_pii_entity(self):
        """Test creating a PII entity."""
        entity = PIIEntity(
            pii_type=PIIType.EMAIL,
            original_value="test@email.com",
            masked_value="[EMAIL REDACTED]",
            start_index=0,
            end_index=14,
            confidence=0.95,
        )

        assert entity.pii_type == PIIType.EMAIL
        assert entity.original_value == "test@email.com"
        assert entity.masked_value == "[EMAIL REDACTED]"


class TestValidationResult:
    """Tests for ValidationResult schema."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_invalid_result_with_errors(self):
        """Test creating an invalid result with errors."""
        errors = [
            ValidationError(
                field_name="date",
                error_type="pattern_mismatch",
                error_message="Invalid date format",
            )
        ]

        result = ValidationResult(is_valid=False, errors=errors)

        assert result.is_valid is False
        assert len(result.errors) == 1


class TestProcessingState:
    """Tests for ProcessingState schema."""

    def test_initial_state(self):
        """Test initial processing state."""
        state = ProcessingState()

        assert state.status == ProcessingStatus.PENDING
        assert state.document_type is None
        assert state.extracted_data is None
        assert state.validation_complete is False
        assert state.redaction_complete is False

    def test_state_with_metadata(self, sample_document_metadata):
        """Test state with document metadata."""
        state = ProcessingState(
            document_metadata=sample_document_metadata,
            raw_content="Test content",
            status=ProcessingStatus.IN_PROGRESS,
        )

        assert state.document_metadata is not None
        assert state.raw_content == "Test content"
        assert state.status == ProcessingStatus.IN_PROGRESS


class TestMetricsReport:
    """Tests for MetricsReport schema."""

    def test_metrics_thresholds(self):
        """Test metrics threshold checking."""
        report = MetricsReport(
            total_documents=100,
            successful_documents=95,
            workflow_success_rate=0.95,
            extraction_accuracy=0.92,
            pii_recall=0.97,
            pii_precision=0.93,
            p95_latency_ms=3500,
        )

        thresholds = report.meets_thresholds()

        assert thresholds["workflow_success"] is True
        assert thresholds["extraction_accuracy"] is True
        assert thresholds["pii_recall"] is True
        assert thresholds["pii_precision"] is True
        assert thresholds["latency_p95"] is True

    def test_metrics_below_thresholds(self):
        """Test metrics below thresholds."""
        report = MetricsReport(
            workflow_success_rate=0.80,
            extraction_accuracy=0.85,
            pii_recall=0.90,
            pii_precision=0.85,
            p95_latency_ms=5000,
        )

        thresholds = report.meets_thresholds()

        assert thresholds["workflow_success"] is False
        assert thresholds["extraction_accuracy"] is False
        assert thresholds["pii_recall"] is False
        assert thresholds["pii_precision"] is False
        assert thresholds["latency_p95"] is False


class TestDocumentSchemas:
    """Tests for document type schemas."""

    def test_invoice_schema_exists(self):
        """Test invoice schema definition."""
        schema = DOCUMENT_SCHEMAS.get(DocumentType.INVOICE)

        assert schema is not None
        assert "required_fields" in schema
        assert "invoice_number" in schema["required_fields"]
        assert "date" in schema["required_fields"]

    def test_receipt_schema_exists(self):
        """Test receipt schema definition."""
        schema = DOCUMENT_SCHEMAS.get(DocumentType.RECEIPT)

        assert schema is not None
        assert "required_fields" in schema
        assert "date" in schema["required_fields"]
        assert "total_amount" in schema["required_fields"]

    def test_all_document_types_have_schemas(self):
        """Test that main document types have schemas."""
        important_types = [
            DocumentType.INVOICE,
            DocumentType.RECEIPT,
            DocumentType.CONTRACT,
            DocumentType.RESUME,
        ]

        for doc_type in important_types:
            assert doc_type in DOCUMENT_SCHEMAS
