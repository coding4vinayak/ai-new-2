# Document Intelligence Agent

AI-powered document intelligence system for automated entity extraction, contract analysis, compliance checking, and batch document processing. Supports fully local, free/open-source model pipelines alongside cloud API options.

## Features

- **Five Extraction Modes**
  - **Local**: Uses spaCy NER and regex patterns for fast, offline entity extraction with no external API dependency.
  - **API**: Leverages OpenAI GPT-4 and Anthropic Claude for high-accuracy extraction with LLM-based understanding.
  - **Hybrid**: Runs local extraction first, then escalates low-confidence fields to API for optimal cost/accuracy balance.
  - **Local LLM**: Uses any OpenAI-compatible local LLM endpoint (Ollama, LM Studio, LocalAI, vLLM) for privacy-preserving extraction without cloud API costs.
  - **Ensemble**: Full pipeline combining OCR ensemble + NER ensemble + optional LayoutLM + optional local LLM. Runs all configured free models together with confidence-weighted result merging for maximum accuracy without any cloud dependencies.

- **Free Model Registry**
  - Central registry tracking all available free/open-source models
  - Health checking and graceful degradation for unavailable models
  - Supports OCR engines (Tesseract, TrOCR, PaddleOCR, DocTR), NER models (spaCy, HuggingFace transformers, LayoutLMv3), and local LLMs

- **OCR Ensemble**
  - Multiple OCR engines with confidence-weighted result merging
  - Engines: Tesseract (default), TrOCR (Microsoft transformer OCR), PaddleOCR, DocTR (Mindee)
  - Similarity-based voting when engines agree; fallback strategies when they disagree

- **NER Ensemble**
  - Multiple NER engines with confidence-weighted entity merging
  - Engines: spaCy (sm/lg), HuggingFace transformer NER (dslim/bert-base-NER), LayoutLMv3
  - Fuzzy matching to group similar entity values across engines

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
+------------------+     +-------------------+     +---------------------+
|   FastAPI REST   |---->|   Orchestrator    |---->|     Extractors      |
|   API Layer      |     |   (DocumentAgent) |     | Local/API/Hybrid/   |
+------------------+     +-------------------+     | LocalLLM/Ensemble   |
        |                         |                +---------------------+
        v                         v                        |
+------------------+     +-------------------+     +---------------------+
|  Action Engine   |     | Industry Modules  |     |  Ensemble Layer     |
|  (Rules/Webhook) |     | Contract/Invoice  |     |  OCR + NER + LLM   |
+------------------+     +-------------------+     +---------------------+
        |                         |                        |
        v                         v               +--------+--------+
+------------------+     +-------------------+    |        |        |
|  Webhook Client  |     |   Audit Logger    |    v        v        v
|  (Dispatch)      |     |   (SQLite)        |  OCR     NER    LayoutLM
+------------------+     +-------------------+  Ensemble Ensemble Engine
                                                  |        |
                                          +-------+--+  +--+-------+
                                          |Tesseract |  |spaCy     |
                                          |TrOCR     |  |HuggingFace|
                                          |PaddleOCR |  |LayoutLM  |
                                          |DocTR     |  +-----------+
                                          +----------+
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
# Edit .env with your API keys (only needed for API/Hybrid modes)

# Run the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Setting Up Local LLM (Ollama)

For fully private, zero-cost extraction using a local LLM:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (llama3 recommended for entity extraction)
ollama pull llama3

# Ollama runs at http://localhost:11434 by default
# The agent auto-detects it with default configuration

# Alternative: use Mistral for faster extraction
ollama pull mistral

# Update .env to use a different model
LOCAL_LLM_MODEL=mistral
```

For LM Studio:
```bash
# Download LM Studio from https://lmstudio.ai
# Load a model (e.g., Mistral-7B-Instruct) and start the server
# Default endpoint: http://localhost:1234/v1

# Update .env
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
LOCAL_LLM_MODEL=local-model
```

For LocalAI:
```bash
# Run with Docker
docker run -p 8080:8080 localai/localai:latest

# Update .env
LOCAL_LLM_BASE_URL=http://localhost:8080/v1
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

### Extract Entities (Local LLM Mode)

```bash
# Requires a running Ollama/LM Studio/LocalAI endpoint
curl -X POST http://localhost:8000/extract/local-llm \
  -F "file=@document.pdf" \
  -G -d "document_type=contract"
```

### Extract Entities (Ensemble Mode)

```bash
# Full pipeline: OCR + NER + optional LayoutLM + optional LLM
curl -X POST http://localhost:8000/extract/ensemble \
  -F "file=@document.pdf" \
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
- Local LLM settings (base_url, model, provider)
- Free model configuration (OCR engines, NER models, layout models)
- Ensemble settings (which engines to combine, voting strategy, thresholds)
- Supported file types

### Ensemble Configuration

The ensemble section in `config/settings.yaml` controls which engines participate:

```yaml
ensemble:
  # OCR engines: tesseract, trocr, paddleocr, doctr
  ocr_engines:
    - tesseract
  # NER models: spacy_sm, spacy_lg, huggingface_ner
  ner_models:
    - spacy_sm
  # Merge strategy: confidence_weighted or majority
  voting_strategy: confidence_weighted
  # Minimum confidence for including entities
  confidence_threshold: 0.6
  # Enable LayoutLM for structured documents (forms, invoices)
  use_layoutlm: false
  # Enable local LLM as additional extraction source
  use_local_llm: false
```

### Free Models Configuration

The `free_models` section configures individual model options:

```yaml
free_models:
  ocr_engines:
    - name: tesseract
      enabled: true
      config:
        lang: eng
    - name: doctr
      enabled: false
      config:
        det_arch: db_resnet50
        reco_arch: crnn_vgg16_bn
    - name: paddleocr
      enabled: false
      config:
        lang: en
  ner_models:
    - name: spacy_sm
      enabled: true
      config:
        model: en_core_web_sm
    - name: spacy_lg
      enabled: false
      config:
        model: en_core_web_lg
  layout_models:
    - name: doctr_layout
      enabled: false
```

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
| `LOCAL_LLM_BASE_URL` | Base URL for local LLM endpoint | Default: http://localhost:11434/v1 |
| `LOCAL_LLM_MODEL` | Model name for local LLM | Default: llama3 |
| `LOCAL_LLM_API_KEY` | API key for local LLM (if required) | Default: not-needed |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov=api -v

# Run specific test file
python -m pytest tests/test_contract_analyzer.py -v

# Run ensemble tests
python -m pytest tests/test_ensemble_extractor.py -v
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
    extractors/          # Extraction modes
      base.py            # BaseExtractor abstract class
      local_extractor.py # spaCy + regex (local mode)
      api_extractor.py   # OpenAI/Claude (API mode)
      hybrid_extractor.py# Local-first with API fallback
      openai_compat_extractor.py  # Local LLM mode
      ensemble_extractor.py       # Full ensemble pipeline
      ner_ensemble.py    # NER engine orchestrator
      ner_engines/       # Individual NER engines
        spacy_ner.py     # spaCy NER engine
        huggingface_ner.py # HuggingFace NER engine
        layoutlm_engine.py # LayoutLMv3 engine
    processors/          # Document format processors
      pdf_processor.py   # PDF text extraction
      image_processor.py # Image preprocessing
      docx_processor.py  # DOCX processing
      ocr_engine.py      # Legacy OCR (Tesseract)
      ocr_ensemble.py    # OCR engine orchestrator
      ocr_engines/       # Individual OCR engines
        tesseract_ocr.py # Tesseract wrapper
        trocr_engine.py  # Microsoft TrOCR
        paddle_ocr_engine.py # PaddleOCR
        doctr_engine.py  # DocTR (Mindee)
    models/              # Pydantic data models
      model_registry.py  # Free model registry
      confidence.py      # Confidence scoring
      document.py        # Document model
      extraction_result.py # ExtractionResult and modes
    industry/            # Enterprise features
    batch/               # Batch processing with progress tracking
    utils/               # Config, audit, webhooks, file utilities
  config/                # YAML configuration files
    settings.yaml        # Main application settings
    entities.yaml        # Entity definitions per doc type
    actions.yaml         # Action rules
  tests/                 # pytest test suite
    sample_docs/         # Sample documents for testing
  Dockerfile             # Multi-stage Docker build
  docker-compose.yml     # Docker Compose with app + Redis
  requirements.txt       # Python dependencies
```

## Technology Stack

- **Framework**: FastAPI with async support
- **NLP**: spaCy for local entity recognition, HuggingFace transformers for advanced NER
- **LLMs**: OpenAI GPT-4, Anthropic Claude (cloud); Ollama, LM Studio, LocalAI (local)
- **OCR**: Tesseract, TrOCR, PaddleOCR, DocTR with ensemble voting
- **Layout Understanding**: LayoutLMv3 for structure-aware extraction
- **Document Processing**: PyPDF2, pdfplumber, python-docx, Pillow
- **Data Validation**: Pydantic v2
- **Database**: SQLite for audit logging and progress tracking
- **Configuration**: YAML files with pydantic-settings
- **Testing**: pytest with pytest-asyncio
- **Containerization**: Docker with multi-stage builds
