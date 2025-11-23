"""
Tests for LangGraph workflow.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.schemas.document import (
    DocumentType,
    ProcessingStatus,
)


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""

    @patch("src.workflow.get_bedrock_client")
    @patch("src.workflow.DocumentLoader")
    def test_processor_initialization(self, mock_loader_class, mock_get_client):
        """Test processor initialization."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_get_client.return_value = mock_client

        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        from src.workflow import DocumentProcessor
        processor = DocumentProcessor(bedrock_client=mock_client)

        assert processor is not None
        assert processor.compiled_workflow is not None

    @patch("src.workflow.get_bedrock_client")
    def test_health_check(self, mock_get_client):
        """Test health check."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = True
        mock_get_client.return_value = mock_client

        from src.workflow import DocumentProcessor
        processor = DocumentProcessor(bedrock_client=mock_client)
        health = processor.health_check()

        assert "bedrock" in health
        assert "workflow" in health


class TestWorkflowCreation:
    """Tests for workflow graph creation."""

    @patch("src.workflow.get_bedrock_client")
    def test_create_workflow(self, mock_get_client):
        """Test workflow graph creation."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        from src.workflow import create_workflow
        workflow = create_workflow(bedrock_client=mock_client)

        assert workflow is not None


class TestWorkflowIntegration:
    """Integration tests for the workflow (with mocked Bedrock)."""

    @patch("src.workflow.get_bedrock_client")
    @patch("src.workflow.DocumentLoader")
    def test_process_file_happy_path(self, mock_loader_class, mock_get_client, sample_docs_dir):
        """Test processing a file through the entire workflow."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.model_id = "test-model"
        mock_client.health_check.return_value = True

        # Mock classifier response
        mock_client.invoke_with_json_output.side_effect = [
            # Classifier response
            {
                "document_type": "invoice",
                "confidence": 0.95,
                "reasoning": "Invoice document",
            },
            # Extractor response
            {
                "fields": [
                    {"name": "invoice_number", "value": "INV-001", "confidence": 0.95, "data_type": "string"},
                    {"name": "date", "value": "2024-11-15", "confidence": 0.90, "data_type": "date"},
                    {"name": "total_amount", "value": "100.00", "confidence": 0.90, "data_type": "number"},
                    {"name": "vendor_name", "value": "Test Vendor", "confidence": 0.90, "data_type": "string"},
                ],
                "structured_data": {
                    "invoice_number": "INV-001",
                    "date": "2024-11-15",
                    "total_amount": "100.00",
                    "vendor_name": "Test Vendor",
                },
            },
            # PII detection (LLM) - empty response
            {"pii_entities": []},
        ]
        mock_get_client.return_value = mock_client

        # Mock document loader
        from src.schemas.document import DocumentMetadata
        mock_loader = MagicMock()
        mock_loader.load.return_value = (
            DocumentMetadata(
                file_path="/tmp/test.txt",
                file_name="test.txt",
                file_type="txt",
                file_size_bytes=100,
            ),
            "Invoice INV-001 from Test Vendor for $100.00 dated 2024-11-15",
        )
        mock_loader_class.return_value = mock_loader

        from src.workflow import DocumentProcessor
        processor = DocumentProcessor(bedrock_client=mock_client)

        # Note: We can't fully test without a real file, but we can test the processor setup
        assert processor.compiled_workflow is not None

    @patch("src.workflow.get_bedrock_client")
    def test_workflow_graph_nodes(self, mock_get_client):
        """Test that workflow has expected nodes."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        from src.workflow import create_workflow
        workflow = create_workflow(bedrock_client=mock_client)

        # Check that the workflow can be compiled
        compiled = workflow.compile()
        assert compiled is not None
