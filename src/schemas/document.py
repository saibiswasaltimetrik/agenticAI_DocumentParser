"""
Pydantic schemas for document processing pipeline.

Defines all data models used for document classification, extraction,
validation, redaction, and reporting.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    REPORT = "report"
    LETTER = "letter"
    FORM = "form"
    RESUME = "resume"
    ID_DOCUMENT = "id_document"
    MEDICAL_RECORD = "medical_record"
    FINANCIAL_STATEMENT = "financial_statement"
    RESEARCH_PAPER = "research_paper"
    UNKNOWN = "unknown"


class PIIType(str, Enum):
    """Types of Personally Identifiable Information."""

    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    IP_ADDRESS = "ip_address"
    MEDICAL_ID = "medical_id"
    TAX_ID = "tax_id"


class AgentType(str, Enum):
    """Types of agents in the pipeline."""

    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    REDACTOR = "redactor"
    REPORTER = "reporter"
    SELF_REPAIR = "self_repair"


class ProcessingStatus(str, Enum):
    """Status of document processing."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class DocumentMetadata(BaseModel):
    """Metadata about the input document."""

    file_path: str = Field(..., description="Path to the document file")
    file_name: str = Field(..., description="Name of the document file")
    file_type: str = Field(..., description="File extension (pdf, jpg, png, txt, etc.)")
    file_size_bytes: int = Field(..., description="Size of the file in bytes")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Processing timestamp"
    )
    page_count: Optional[int] = Field(None, description="Number of pages (for PDFs)")
    ocr_applied: bool = Field(False, description="Whether OCR was used")
    source_hash: Optional[str] = Field(None, description="SHA256 hash of source file")


class ExtractedField(BaseModel):
    """A single extracted field from a document."""

    name: str = Field(..., description="Field name")
    value: Any = Field(..., description="Extracted value")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0-1)"
    )
    source_location: Optional[str] = Field(
        None, description="Location in document (page, coordinates)"
    )
    data_type: str = Field(default="string", description="Data type of the field")
    normalized_value: Optional[Any] = Field(
        None, description="Normalized/standardized value"
    )


class ExtractedData(BaseModel):
    """All extracted data from a document."""

    document_type: DocumentType = Field(..., description="Classified document type")
    fields: List[ExtractedField] = Field(
        default_factory=list, description="List of extracted fields"
    )
    raw_text: Optional[str] = Field(None, description="Raw extracted text")
    structured_data: Dict[str, Any] = Field(
        default_factory=dict, description="Structured JSON representation"
    )
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert extracted fields to a simple dictionary."""
        return {field.name: field.value for field in self.fields}


class PIIEntity(BaseModel):
    """A detected PII entity."""

    pii_type: PIIType = Field(..., description="Type of PII")
    original_value: str = Field(..., description="Original PII value")
    masked_value: str = Field(..., description="Masked/redacted value")
    start_index: int = Field(..., description="Start position in text")
    end_index: int = Field(..., description="End position in text")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence (0-1)"
    )
    context: Optional[str] = Field(
        None, description="Surrounding context for the PII"
    )


class RedactionResult(BaseModel):
    """Result of PII redaction."""

    original_text: str = Field(..., description="Original text before redaction")
    redacted_text: str = Field(..., description="Text after PII redaction")
    pii_entities: List[PIIEntity] = Field(
        default_factory=list, description="List of detected and redacted PII"
    )
    total_pii_found: int = Field(0, description="Total PII entities found")
    pii_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Count of PII by type"
    )
    redaction_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    """A validation error."""

    field_name: str = Field(..., description="Name of the invalid field")
    error_type: str = Field(..., description="Type of validation error")
    error_message: str = Field(..., description="Human-readable error message")
    expected_value: Optional[Any] = Field(None, description="Expected value or format")
    actual_value: Optional[Any] = Field(None, description="Actual value found")
    severity: str = Field(default="error", description="error, warning, or info")


class ValidationResult(BaseModel):
    """Result of data validation."""

    is_valid: bool = Field(..., description="Whether all validations passed")
    errors: List[ValidationError] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: List[ValidationError] = Field(
        default_factory=list, description="List of validation warnings"
    )
    schema_version: str = Field(default="1.0", description="Schema version used")
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    repair_attempted: bool = Field(
        False, description="Whether self-repair was attempted"
    )
    repair_successful: bool = Field(
        False, description="Whether self-repair succeeded"
    )


class AgentDecision(BaseModel):
    """Record of an agent's decision for responsible AI logging."""

    agent_type: AgentType = Field(..., description="Type of agent")
    agent_name: str = Field(..., description="Name/identifier of the agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_summary: str = Field(..., description="Summary of input data")
    output_summary: str = Field(..., description="Summary of output/decision")
    confidence: Optional[float] = Field(None, description="Confidence in decision")
    reasoning: Optional[str] = Field(None, description="Explanation of decision")
    duration_ms: float = Field(..., description="Processing duration in milliseconds")
    model_used: Optional[str] = Field(None, description="LLM model used")
    tokens_used: Optional[int] = Field(None, description="Number of tokens consumed")
    retry_count: int = Field(0, description="Number of retries needed")
    error_message: Optional[str] = Field(
        None, description="Error message if operation failed"
    )


class ProcessingState(BaseModel):
    """
    State object passed between agents in the LangGraph workflow.

    This is the main state container for the document processing pipeline.
    """

    # Input
    document_metadata: Optional[DocumentMetadata] = None
    raw_content: Optional[str] = None

    # Classification
    document_type: Optional[DocumentType] = None
    classification_confidence: float = 0.0

    # Extraction
    extracted_data: Optional[ExtractedData] = None
    extraction_complete: bool = False

    # Validation
    validation_result: Optional[ValidationResult] = None
    validation_complete: bool = False
    needs_repair: bool = False
    repair_attempts: int = 0
    max_repair_attempts: int = 3

    # Redaction
    redaction_result: Optional[RedactionResult] = None
    redaction_complete: bool = False

    # Processing metadata
    status: ProcessingStatus = ProcessingStatus.PENDING
    current_agent: Optional[AgentType] = None
    agent_decisions: List[AgentDecision] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    class Config:
        use_enum_values = True


class MetricsReport(BaseModel):
    """Metrics report for processing evaluation."""

    # Processing metrics
    total_documents: int = Field(0, description="Total documents processed")
    successful_documents: int = Field(0, description="Successfully processed documents")
    failed_documents: int = Field(0, description="Failed documents")
    workflow_success_rate: float = Field(0.0, description="Success rate (0-1)")

    # Extraction metrics
    extraction_accuracy: float = Field(0.0, description="Field extraction accuracy")
    fields_extracted: int = Field(0, description="Total fields extracted")
    fields_correct: int = Field(0, description="Fields with correct values")

    # PII metrics
    pii_total_detected: int = Field(0, description="Total PII entities detected")
    pii_true_positives: int = Field(0, description="Correctly identified PII")
    pii_false_positives: int = Field(0, description="Incorrectly flagged as PII")
    pii_false_negatives: int = Field(0, description="Missed PII (if ground truth known)")
    pii_recall: float = Field(0.0, description="PII recall (sensitivity)")
    pii_precision: float = Field(0.0, description="PII precision")
    pii_f1_score: float = Field(0.0, description="PII F1 score")

    # Latency metrics
    avg_latency_ms: float = Field(0.0, description="Average processing latency")
    p50_latency_ms: float = Field(0.0, description="P50 latency")
    p95_latency_ms: float = Field(0.0, description="P95 latency")
    p99_latency_ms: float = Field(0.0, description="P99 latency")

    # Agent-specific metrics
    classifier_accuracy: float = Field(0.0, description="Classification accuracy")
    validator_pass_rate: float = Field(0.0, description="Initial validation pass rate")
    repair_success_rate: float = Field(0.0, description="Self-repair success rate")

    # Report metadata
    report_timestamp: datetime = Field(default_factory=datetime.utcnow)
    evaluation_period_start: Optional[datetime] = None
    evaluation_period_end: Optional[datetime] = None

    def meets_thresholds(
        self,
        extraction_threshold: float = 0.90,
        pii_recall_threshold: float = 0.95,
        pii_precision_threshold: float = 0.90,
        workflow_threshold: float = 0.90,
        latency_threshold_ms: float = 4000,
    ) -> Dict[str, bool]:
        """Check if metrics meet the required thresholds."""
        return {
            "extraction_accuracy": self.extraction_accuracy >= extraction_threshold,
            "pii_recall": self.pii_recall >= pii_recall_threshold,
            "pii_precision": self.pii_precision >= pii_precision_threshold,
            "workflow_success": self.workflow_success_rate >= workflow_threshold,
            "latency_p95": self.p95_latency_ms <= latency_threshold_ms,
        }


class ProcessingResult(BaseModel):
    """Final result of document processing."""

    success: bool = Field(..., description="Whether processing succeeded")
    document_metadata: DocumentMetadata = Field(..., description="Document metadata")
    document_type: DocumentType = Field(..., description="Classified document type")
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted and validated data"
    )
    redacted_text: Optional[str] = Field(None, description="Redacted text content")
    pii_summary: Dict[str, int] = Field(
        default_factory=dict, description="Summary of redacted PII by type"
    )
    validation_passed: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    processing_time_ms: float = Field(..., description="Total processing time")
    agent_trace: List[AgentDecision] = Field(
        default_factory=list, description="Trace of agent decisions"
    )


# Schema definitions for different document types (for validation)
DOCUMENT_SCHEMAS: Dict[DocumentType, Dict[str, Any]] = {
    DocumentType.INVOICE: {
        "required_fields": [
            "invoice_number",
            "date",
            "total_amount",
            "vendor_name",
        ],
        "optional_fields": [
            "line_items",
            "tax_amount",
            "due_date",
            "payment_terms",
            "billing_address",
            "shipping_address",
        ],
        "field_patterns": {
            "invoice_number": r"^[A-Z0-9-]+$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "total_amount": r"^\d+\.?\d*$",
        },
    },
    DocumentType.RECEIPT: {
        "required_fields": ["date", "total_amount", "merchant_name"],
        "optional_fields": ["items", "payment_method", "tax_amount", "subtotal"],
        "field_patterns": {
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "total_amount": r"^\d+\.?\d*$",
        },
    },
    DocumentType.CONTRACT: {
        "required_fields": ["parties", "effective_date", "terms"],
        "optional_fields": [
            "termination_date",
            "signatures",
            "governing_law",
            "amendments",
        ],
        "field_patterns": {
            "effective_date": r"^\d{4}-\d{2}-\d{2}$",
        },
    },
    DocumentType.RESUME: {
        "required_fields": ["name", "contact_info"],
        "optional_fields": [
            "education",
            "experience",
            "skills",
            "certifications",
            "summary",
        ],
        "field_patterns": {},
    },
    DocumentType.ID_DOCUMENT: {
        "required_fields": ["name", "id_number", "issue_date"],
        "optional_fields": [
            "expiry_date",
            "date_of_birth",
            "address",
            "photo",
            "issuing_authority",
        ],
        "field_patterns": {},
    },
    DocumentType.MEDICAL_RECORD: {
        "required_fields": ["patient_name", "date", "provider"],
        "optional_fields": [
            "diagnosis",
            "medications",
            "procedures",
            "notes",
            "lab_results",
        ],
        "field_patterns": {},
    },
    DocumentType.FINANCIAL_STATEMENT: {
        "required_fields": ["period", "entity_name", "statement_type"],
        "optional_fields": [
            "revenue",
            "expenses",
            "net_income",
            "assets",
            "liabilities",
            "equity",
        ],
        "field_patterns": {},
    },
    DocumentType.RESEARCH_PAPER: {
        "required_fields": ["title", "authors"],
        "optional_fields": [
            "abstract",
            "keywords",
            "institution",
            "publication_date",
            "doi",
            "references",
        ],
        "field_patterns": {},
    },
}
