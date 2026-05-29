# Document Intelligence Agent

AI-powered document intelligence system for automated entity extraction, contract analysis, compliance checking, and batch document processing.

## Features

- **Three Extraction Modes**
  - **Local**: Uses spaCy NER and regex patterns for fast, offline entity extraction with no external API dependency.
  - **API**: Leverages OpenAI GPT-4 and Anthropic Claude for high-accuracy extraction with LLM-based understanding.
  - **Hybrid**: Runs local extraction first, then escalates low-confidence fields to API for optimal cost/accuracy balance.

- **Industry Modules**
  - Contract analysis with clause extraction, obligation detection, and risk scoring
  - Invoice processing with line item extraction and amount validation
  - Compliance checking (GDPR, contract requirements)
  - PII detection and redaction
  - Document classification
  - Version comparison

- **Batch Processing** - Process multiple documents concurrently with progress tracking
- **Action Engine** - Configurable rules that trigger webhooks, notifications, or routing based on extraction results
- **Audit Logging** - Full audit trail of all extraction operations

## Architecture

```
+------------------+     +-------------------+     +-----------------+
|   FastAPI REST   |---->|   Orchestrator    |---->|   Extractors    |
|   API Layer      |     |   (DocumentAgent) |     | Local/API/Hybrid|
+------------------+     +-------------------+     +-----------------+
        |                         |                        |
        v                         v                        v
+------------------+     +-------------------+     +-----------------+
|  Action Engine   |     | Industry Modules  |     |   Processors    |
|  (Rules/Webhook) |     | Contract/Invoice  |     |  PDF/DOCX/Image |
+------------------+     +-------------------+     +-----------------+
        |                         |
        v                         v
+------------------+     +-------------------+
|  Webhook Client  |     |   Audit Logger    |
|  (Dispatch)      |     |   (SQLite)        |
+------------------+     +-------------------+
```

## Quick Start

### Using Docker (Recommended)

```bash
# Copy environment file and add your API keys
cp .env.example .env

# Build and run
docker-compose up --build

# The API is available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Local Development

```bash
# Install Python 3.11
pyenv install 3.11.14
pyenv shell 3.11.14

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, visit `/docs` for the interactive Swagger UI.

### Health Check

```bash
curl http://localhost:8000/health
```

### Extract Entities (Local Mode)

```bash
curl -X POST http://localhost:8000/extract/local \
  -F "file=@document.pdf"
```

### Extract Entities (API Mode)

```bash
curl -X POST http://localhost:8000/extract/api \
  -F "file=@contract.docx" \
  -G -d "document_type=contract"
```

### Extract Entities (Hybrid Mode)

```bash
curl -X POST http://localhost:8000/extract/hybrid \
  -F "file=@invoice.pdf" \
  -G -d "document_type=invoice"
```

### Auto-Detect Mode

```bash
curl -X POST http://localhost:8000/extract/auto \
  -F "file=@document.txt"
```

### Batch Processing

```bash
# Start batch job
curl -X POST http://localhost:8000/batch/process \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.txt" \
  -G -d "mode=hybrid"

# Check status
curl http://localhost:8000/batch/status/{job_id}

# Get results
curl http://localhost:8000/batch/result/{job_id}
```

### Action Rules

```bash
# List rules
curl http://localhost:8000/actions/rules

# Create a rule
curl -X POST http://localhost:8000/actions/rules \
  -H "Content-Type: application/json" \
  -d '{"name": "high_value_alert", "trigger_condition": "total_amount > 100000", "action_type": "webhook", "action_endpoint": "https://hooks.example.com/alert"}'

# Evaluate rules manually
curl -X POST http://localhost:8000/actions/evaluate \
  -H "Content-Type: application/json" \
  -d '{"document_id": "test-123", "entities": {"risk_score": 0.9}, "overall_confidence": 0.85}'
```

### Webhooks

```bash
# Register a webhook
curl -X POST http://localhost:8000/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "https://hooks.example.com/events", "events": ["extraction.complete", "action.triggered"]}'

# List webhooks
curl http://localhost:8000/webhooks

# Test a webhook
curl -X POST http://localhost:8000/webhooks/{webhook_id}/test
```

## Configuration

### settings.yaml

Located at `config/settings.yaml`, controls:
- Extraction modes and default mode
- Confidence thresholds for hybrid escalation
- Batch processing limits (max concurrent, max batch size)
- API settings (model, timeout, retries)
- Supported file types

### actions.yaml

Located at `config/actions.yaml`, defines action rules that trigger automatically based on extraction results. Each rule has a trigger condition and an action to execute.

### entities.yaml

Located at `config/entities.yaml`, defines entity schemas for different document types (contracts, invoices, general documents).

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 extraction | For API/Hybrid mode |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude extraction | For API fallback |
| `TESSERACT_PATH` | Path to tesseract binary | For OCR |
| `SPACY_MODEL` | spaCy model name | Default: en_core_web_sm |
| `DATABASE_URL` | SQLite database URL | Default: sqlite:///data/audit.db |
| `WEBHOOK_SECRET` | Secret for webhook HMAC signing | For webhooks |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov=api -v

# Run specific test file
python -m pytest tests/test_contract_analyzer.py -v
```

## Project Structure

```
ai-new-2/
  api/                    # FastAPI REST API layer
    main.py              # App initialization, middleware, routes
    routes/              # API route modules
      health.py          # Health check and metrics
      extract.py         # Document extraction endpoints
      batch.py           # Batch processing endpoints
      actions.py         # Action rules management
      webhooks.py        # Webhook registration and management
  src/                   # Core application logic
    agent/               # Orchestrator and action engine
    extractors/          # Three extraction modes (Local, API, Hybrid)
    processors/          # Document format processors (PDF, Image, DOCX, OCR)
    models/              # Pydantic data models
    industry/            # Enterprise features
    batch/               # Batch processing with progress tracking
    utils/               # Config, audit, webhooks, file utilities
  config/                # YAML configuration files
  tests/                 # pytest test suite
    sample_docs/         # Sample documents for testing
  Dockerfile             # Multi-stage Docker build
  docker-compose.yml     # Docker Compose with app + Redis
  requirements.txt       # Python dependencies
```

## Technology Stack

- **Framework**: FastAPI with async support
- **NLP**: spaCy for local entity recognition
- **LLMs**: OpenAI GPT-4, Anthropic Claude for API extraction
- **OCR**: Tesseract via pytesseract
- **Document Processing**: PyPDF2, pdfplumber, python-docx, Pillow
- **Data Validation**: Pydantic v2
- **Database**: SQLite for audit logging and progress tracking
- **Configuration**: YAML files with pydantic-settings
- **Testing**: pytest with pytest-asyncio
- **Containerization**: Docker with multi-stage builds
