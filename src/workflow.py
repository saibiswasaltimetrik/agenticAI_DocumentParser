"""
LangGraph workflow for document processing pipeline.

Orchestrates the flow between ClassifierAgent, ExtractorAgent, ValidatorAgent,
RedactorAgent, and ReporterAgent using LangGraph state management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .agents import (
    ClassifierAgent,
    ExtractorAgent,
    ValidatorAgent,
    RedactorAgent,
    ReporterAgent,
)
from .core.bedrock_client import BedrockClient, get_bedrock_client
from .schemas.document import (
    AgentType,
    DocumentMetadata,
    ProcessingResult,
    ProcessingState,
    ProcessingStatus,
)
from .utils.document_loader import DocumentLoader
from .utils.logging import ResponsibleAILogger, get_logger

logger = get_logger(__name__)


class WorkflowState(TypedDict, total=False):
    """TypedDict for LangGraph state."""

    state: ProcessingState


def create_workflow(
    bedrock_client: Optional[BedrockClient] = None,
    rai_logger: Optional[ResponsibleAILogger] = None,
) -> StateGraph:
    """
    Create the document processing workflow graph.

    Args:
        bedrock_client: Optional Bedrock client
        rai_logger: Optional Responsible AI logger

    Returns:
        Configured StateGraph
    """
    client = bedrock_client or get_bedrock_client()

    # Initialize agents
    classifier = ClassifierAgent(bedrock_client=client, rai_logger=rai_logger)
    extractor = ExtractorAgent(bedrock_client=client, rai_logger=rai_logger)
    validator = ValidatorAgent(bedrock_client=client, rai_logger=rai_logger)
    redactor = RedactorAgent(bedrock_client=client, rai_logger=rai_logger)
    reporter = ReporterAgent(rai_logger=rai_logger)

    # Define node functions
    def classify_node(state: WorkflowState) -> WorkflowState:
        """Classify document type."""
        logger.info("Executing classify node")
        processing_state = state["state"]
        processing_state = classifier.process(processing_state)
        return {"state": processing_state}

    def extract_node(state: WorkflowState) -> WorkflowState:
        """Extract data from document."""
        logger.info("Executing extract node")
        processing_state = state["state"]
        processing_state = extractor.process(processing_state)
        return {"state": processing_state}

    def validate_node(state: WorkflowState) -> WorkflowState:
        """Validate extracted data."""
        logger.info("Executing validate node")
        processing_state = state["state"]
        processing_state = validator.process(processing_state)
        return {"state": processing_state}

    def redact_node(state: WorkflowState) -> WorkflowState:
        """Redact PII from document."""
        logger.info("Executing redact node")
        processing_state = state["state"]
        processing_state = redactor.process(processing_state)
        return {"state": processing_state}

    def report_node(state: WorkflowState) -> WorkflowState:
        """Generate final report."""
        logger.info("Executing report node")
        processing_state = state["state"]
        processing_state = reporter.process(processing_state)
        return {"state": processing_state}

    def should_repair(state: WorkflowState) -> str:
        """
        Conditional edge: check if repair is needed.

        Returns:
            "repair" if repair needed, "redact" otherwise
        """
        processing_state = state["state"]
        if processing_state.needs_repair and processing_state.repair_attempts < processing_state.max_repair_attempts:
            logger.info("Validation failed, routing to repair")
            return "repair"
        return "redact"

    # Build the graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("redact", redact_node)
    workflow.add_node("report", report_node)

    # Add edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "extract")
    workflow.add_edge("extract", "validate")

    # Conditional edge for repair loop
    workflow.add_conditional_edges(
        "validate",
        should_repair,
        {
            "repair": "validate",  # Loop back to validate (which includes repair)
            "redact": "redact",
        },
    )

    workflow.add_edge("redact", "report")
    workflow.add_edge("report", END)

    return workflow


class DocumentProcessor:
    """
    Main document processor class using LangGraph workflow.

    Provides high-level interface for processing documents through
    the agentic pipeline.
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        rai_logger: Optional[ResponsibleAILogger] = None,
    ):
        """
        Initialize the document processor.

        Args:
            bedrock_client: Optional Bedrock client
            rai_logger: Optional Responsible AI logger
        """
        self.bedrock_client = bedrock_client or get_bedrock_client()
        self.rai_logger = rai_logger or ResponsibleAILogger()
        self.document_loader = DocumentLoader()
        self.reporter = ReporterAgent(rai_logger=self.rai_logger)

        # Create and compile workflow
        self.workflow = create_workflow(
            bedrock_client=self.bedrock_client,
            rai_logger=self.rai_logger,
        )
        self.compiled_workflow = self.workflow.compile()

    def process_file(self, file_path: str) -> ProcessingResult:
        """
        Process a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            ProcessingResult with all outputs
        """
        logger.info(f"Processing file: {file_path}")

        # Load document
        metadata, content = self.document_loader.load(file_path)

        # Initialize state
        initial_state = ProcessingState(
            document_metadata=metadata,
            raw_content=content,
            status=ProcessingStatus.IN_PROGRESS,
            start_time=datetime.utcnow(),
        )

        # Run workflow
        workflow_state: WorkflowState = {"state": initial_state}

        try:
            final_state = self.compiled_workflow.invoke(workflow_state)
            processing_state = final_state["state"]
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state.status = ProcessingStatus.FAILED
            initial_state.errors.append(str(e))
            initial_state.end_time = datetime.utcnow()
            processing_state = initial_state

        # Generate result
        result = self.reporter.generate_result(processing_state)

        # Save result
        self.reporter.save_result(result)

        # Save RAI log
        self.rai_logger.save()

        return result

    def process_batch(self, file_paths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple documents.

        Args:
            file_paths: List of file paths

        Returns:
            List of ProcessingResults
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Create error result
                from pathlib import Path

                path = Path(file_path)
                error_result = ProcessingResult(
                    success=False,
                    document_metadata=DocumentMetadata(
                        file_path=file_path,
                        file_name=path.name,
                        file_type=path.suffix.lstrip("."),
                        file_size_bytes=0,
                    ),
                    document_type=None,
                    extracted_data={},
                    validation_passed=False,
                    errors=[str(e)],
                    processing_time_ms=0,
                )
                results.append(error_result)

        # Generate metrics report
        self.reporter.generate_metrics_report(results, "batch_metrics.json")

        return results

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.

        Returns:
            Dictionary with component health status
        """
        return {
            "bedrock": self.bedrock_client.health_check(),
            "workflow": self.compiled_workflow is not None,
        }


def process_document(file_path: str) -> ProcessingResult:
    """
    Convenience function to process a single document.

    Args:
        file_path: Path to the document file

    Returns:
        ProcessingResult
    """
    processor = DocumentProcessor()
    return processor.process_file(file_path)
