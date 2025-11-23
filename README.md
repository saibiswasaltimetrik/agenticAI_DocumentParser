# Agentic Document Processor

A LangGraph-based document processing pipeline that performs classification, extraction, validation, and PII redaction using Amazon Bedrock and Mistral Large.

## Features

- **Document Classification**: Automatically classify documents (invoice, receipt, contract, resume, etc.)
- **Data Extraction**: Extract key fields from documents with confidence scores
- **Validation & Self-Repair**: JSON-Schema validation with automatic repair attempts
- **PII Redaction**: Detect and mask personally identifiable information
- **Responsible AI Logging**: Complete audit trail of agent decisions
- **FastAPI REST API**: HTTP endpoints for document processing
- **CLI Interface**: Command-line tool for batch processing

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ClassifierAgent │ ──▶ │  ExtractorAgent │ ──▶ │  ValidatorAgent │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌──────────────────────────┴──────┐
                              │           Repair Loop           │
                              └──────────────────────────┬──────┘
                                                         │
                        ┌─────────────────┐     ┌────────▼────────┐
                        │  ReporterAgent  │ ◀── │  RedactorAgent  │
                        └─────────────────┘     └─────────────────┘
```

## Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Amazon Bedrock (Mistral Large 2402)
- **Framework**: LangChain
- **API**: FastAPI
- **Validation**: Pydantic
- **Retry Logic**: Tenacity
- **OCR**: PyTesseract (optional)
- **Testing**: Pytest

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd agenticAI_DocumentParser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your AWS credentials and preferences:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
BEDROCK_MODEL_ID=mistral.mistral-large-2402-v1:0
```

## Usage

### CLI

```bash
# Process a single document
python main.py process /path/to/document.pdf

# Process multiple documents
python main.py batch /path/to/doc1.pdf /path/to/doc2.txt

# Get output as JSON
python main.py process /path/to/document.pdf --format json

# Check health status
python main.py health

# List supported document types
python main.py document-types
```

### API Server

```bash
# Start the FastAPI server
python run_server.py

# Or use the CLI
python main.py serve --host 0.0.0.0 --port 8000
```

API Endpoints:

- `GET /health` - Health check
- `POST /process` - Process document by file path
- `POST /process/upload` - Upload and process document
- `POST /process/batch` - Process multiple documents
- `GET /metrics` - Get processing metrics
- `GET /document-types` - List supported document types

Example API request:

```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/invoice.pdf"}'
```

### Programmatic Usage

```python
from src.workflow import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process a document
result = processor.process_file("/path/to/document.pdf")

# Access results
print(f"Document Type: {result.document_type}")
print(f"Extracted Data: {result.extracted_data}")
print(f"PII Found: {result.pii_summary}")
print(f"Validation Passed: {result.validation_passed}")
```

## Evaluation Metrics

The pipeline targets the following performance thresholds:

| Metric | Target |
|--------|--------|
| Extraction Accuracy | >= 90% |
| PII Recall | >= 95% |
| PII Precision | >= 90% |
| Workflow Success Rate | >= 90% |
| P95 Latency | <= 4s per document |

## Supported Document Types

- Invoice
- Receipt
- Contract
- Report
- Letter
- Form
- Resume
- ID Document
- Medical Record
- Financial Statement
- Research Paper

## PII Types Detected

- Social Security Numbers (SSN)
- Email addresses
- Phone numbers
- Credit card numbers
- Bank account numbers
- Addresses
- Names
- Dates of birth
- Passport numbers
- Driver's license numbers
- IP addresses
- Medical IDs
- Tax IDs

## Project Structure

```
agenticAI_DocumentParser/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py           # Base agent class
│   │   ├── classifier.py     # Document classification
│   │   ├── extractor.py      # Data extraction
│   │   ├── validator.py      # Validation & self-repair
│   │   ├── redactor.py       # PII detection & redaction
│   │   └── reporter.py       # Metrics & reporting
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration settings
│   │   └── bedrock_client.py # AWS Bedrock client
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── document.py       # Pydantic schemas
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py        # Logging utilities
│   │   └── document_loader.py # Document loading
│   ├── workflow.py           # LangGraph workflow
│   ├── api.py                # FastAPI endpoints
│   └── cli.py                # CLI interface
├── tests/
│   ├── conftest.py           # Test fixtures
│   ├── test_schemas.py       # Schema tests
│   ├── test_agents.py        # Agent tests
│   ├── test_workflow.py      # Workflow tests
│   ├── test_api.py           # API tests
│   └── test_edge_cases.py    # Edge case tests
├── sample_docs/              # Sample test documents
├── output/                   # Processing output
├── logs/                     # Application logs
├── main.py                   # Main entry point
├── run_server.py             # Server script
├── requirements.txt          # Dependencies
├── pytest.ini                # Pytest configuration
├── pyproject.toml            # Project configuration
├── .env.example              # Environment template
└── README.md
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_agents.py

# Run specific test class
pytest tests/test_agents.py::TestRedactorAgent

# Skip slow tests
pytest -m "not slow"
```

## Responsible AI

The pipeline includes comprehensive logging for responsible AI compliance:

- **Decision Logging**: Every agent decision is logged with input/output summaries
- **PII Metrics**: Precision, recall, and F1 scores for PII detection
- **Audit Trail**: Complete trace of document processing flow
- **Error Tracking**: All errors and retry attempts are logged

Logs are saved to `logs/responsible_ai_log.json`.

## Error Handling

- **Retry Logic**: Automatic retry with exponential backoff for transient errors
- **Timeout Handling**: Graceful handling of Bedrock timeouts
- **Validation Repair**: Automatic attempts to fix validation errors
- **Graceful Degradation**: Processing continues even if individual steps fail

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
