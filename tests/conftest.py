"""
Pytest configuration and fixtures for testing.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.document import (
    DocumentMetadata,
    DocumentType,
    ExtractedData,
    ExtractedField,
    ProcessingState,
    ProcessingStatus,
    RedactionResult,
    ValidationResult,
)


@pytest.fixture
def sample_invoice_content():
    """Sample invoice content for testing."""
    return """
    INVOICE
    Invoice Number: INV-2024-001234
    Date: 2024-11-15

    Bill To:
    John Smith
    Email: john.smith@email.com
    Phone: (555) 123-4567

    Total: $2,495.50

    SSN: 123-45-6789
    Credit Card: 4532-1234-5678-9012
    """


@pytest.fixture
def sample_receipt_content():
    """Sample receipt content for testing."""
    return """
    GROCERY MART
    Receipt #45678
    Date: 2024-11-20

    Items:
    - Organic Milk: $5.99
    - Bread: $3.49

    Total: $49.96

    Customer: Jane Doe
    Email: jane.doe@gmail.com
    """


@pytest.fixture
def sample_document_metadata():
    """Sample document metadata fixture."""
    return DocumentMetadata(
        file_path="/tmp/test_document.txt",
        file_name="test_document.txt",
        file_type="txt",
        file_size_bytes=1024,
    )


@pytest.fixture
def sample_processing_state(sample_document_metadata, sample_invoice_content):
    """Sample processing state fixture."""
    return ProcessingState(
        document_metadata=sample_document_metadata,
        raw_content=sample_invoice_content,
        status=ProcessingStatus.PENDING,
    )


@pytest.fixture
def sample_extracted_data():
    """Sample extracted data fixture."""
    return ExtractedData(
        document_type=DocumentType.INVOICE,
        fields=[
            ExtractedField(name="invoice_number", value="INV-2024-001234", confidence=0.95),
            ExtractedField(name="date", value="2024-11-15", confidence=0.92),
            ExtractedField(name="total_amount", value="2495.50", confidence=0.90),
            ExtractedField(name="vendor_name", value="ABC Company", confidence=0.88),
        ],
        structured_data={
            "invoice_number": "INV-2024-001234",
            "date": "2024-11-15",
            "total_amount": "2495.50",
            "vendor_name": "ABC Company",
        },
    )


@pytest.fixture
def mock_bedrock_client():
    """Mock Bedrock client for testing without AWS."""
    with patch("src.core.bedrock_client.BedrockClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "healthy"
        mock_instance.invoke_with_json_output.return_value = {
            "document_type": "invoice",
            "confidence": 0.95,
            "reasoning": "Document contains invoice number and billing information",
        }
        mock_instance.model_id = "mistral.mistral-large-2402-v1:0"
        mock_instance.health_check.return_value = True
        MockClient.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_bedrock_extractor_response():
    """Mock response for extractor agent."""
    return {
        "fields": [
            {"name": "invoice_number", "value": "INV-2024-001234", "confidence": 0.95, "data_type": "string"},
            {"name": "date", "value": "2024-11-15", "confidence": 0.92, "data_type": "date"},
            {"name": "total_amount", "value": "2495.50", "confidence": 0.90, "data_type": "number"},
            {"name": "vendor_name", "value": "ABC Company", "confidence": 0.88, "data_type": "string"},
        ],
        "structured_data": {
            "invoice_number": "INV-2024-001234",
            "date": "2024-11-15",
            "total_amount": "2495.50",
            "vendor_name": "ABC Company",
        },
    }


@pytest.fixture
def sample_docs_dir():
    """Path to sample documents directory."""
    return Path(__file__).parent.parent / "sample_docs"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def noisy_ocr_content():
    """Content simulating OCR noise."""
    return """
    1NVOICE
    lnv0ice Numb3r: lNV-2O24-OO1234
    Dat3: 2024-ll-l5

    BiII To:
    J0hn Smlth
    Ema1l: john.smlth@emall.com
    Ph0ne: (555) l23-4567

    T0tal: $2,495.5O
    """


@pytest.fixture
def missing_fields_content():
    """Content with missing required fields."""
    return """
    INVOICE

    Some random text without proper invoice structure.
    No invoice number, no date, no amounts.

    Email: test@test.com
    """
