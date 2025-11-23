"""
FastAPI REST API for the document processing pipeline.
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .core.config import settings
from .schemas.document import (
    DocumentType,
    MetricsReport,
    ProcessingResult,
)
from .workflow import DocumentProcessor
from .utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Document Processor",
    description="LangGraph-based document processing pipeline with classification, extraction, validation, and PII redaction.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance (lazy initialization)
_processor: Optional[DocumentProcessor] = None


def get_processor() -> DocumentProcessor:
    """Get or create the document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


# Request/Response models
class ProcessRequest(BaseModel):
    """Request model for processing a file by path."""
    file_path: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    components: dict


class ProcessResponse(BaseModel):
    """Response model for document processing."""
    success: bool
    document_type: Optional[str]
    extracted_data: dict
    pii_summary: dict
    validation_passed: bool
    errors: List[str]
    processing_time_ms: float


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of the service and its components.
    """
    processor = get_processor()
    component_health = processor.health_check()

    overall_status = "healthy" if all(component_health.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        components=component_health,
    )


@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Process a document by file path.

    Args:
        request: ProcessRequest with file_path

    Returns:
        ProcessResponse with processing results
    """
    try:
        processor = get_processor()
        result = processor.process_file(request.file_path)

        return ProcessResponse(
            success=result.success,
            document_type=result.document_type.value if result.document_type else None,
            extracted_data=result.extracted_data,
            pii_summary=result.pii_summary,
            validation_passed=result.validation_passed,
            errors=result.errors,
            processing_time_ms=result.processing_time_ms,
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/upload", response_model=ProcessResponse)
async def process_uploaded_document(file: UploadFile = File(...)):
    """
    Process an uploaded document file.

    Args:
        file: Uploaded file

    Returns:
        ProcessResponse with processing results
    """
    try:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            processor = get_processor()
            result = processor.process_file(tmp_path)

            return ProcessResponse(
                success=result.success,
                document_type=result.document_type.value if result.document_type else None,
                extracted_data=result.extracted_data,
                pii_summary=result.pii_summary,
                validation_passed=result.validation_passed,
                errors=result.errors,
                processing_time_ms=result.processing_time_ms,
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/batch")
async def process_batch(request: List[ProcessRequest]):
    """
    Process multiple documents.

    Args:
        request: List of ProcessRequest with file paths

    Returns:
        List of ProcessResponse results
    """
    try:
        processor = get_processor()
        file_paths = [r.file_path for r in request]
        results = processor.process_batch(file_paths)

        responses = []
        for result in results:
            responses.append(ProcessResponse(
                success=result.success,
                document_type=result.document_type.value if result.document_type else None,
                extracted_data=result.extracted_data,
                pii_summary=result.pii_summary,
                validation_passed=result.validation_passed,
                errors=result.errors,
                processing_time_ms=result.processing_time_ms,
            ))

        return responses

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get current metrics from the processor.
    """
    try:
        metrics_path = Path(settings.output_dir) / settings.metrics_file
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                return json.load(f)
        return {"message": "No metrics available yet"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document-types")
async def list_document_types():
    """
    List supported document types.
    """
    return {
        "document_types": [dt.value for dt in DocumentType],
    }


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    """
    Start the FastAPI server.

    Args:
        host: Server host (defaults to settings)
        port: Server port (defaults to settings)
    """
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
