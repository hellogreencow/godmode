# GODMODE 🧠

**A superhuman, ontological Question Foresight Engine**

*"See the questions before you ask them."*

## Overview

GODMODE is an advanced AI system that operates as a Question Foresight Engine, designed to help you explore complex decisions by generating progressive question ladders. It operates in dual modes:

- **Operator**: Silent, parallel enumeration, ranking, stitching, and memory updates
- **Interface**: Crisp chat replies + structured graph/ontology updates for tree UI rendering

## Core Features

### 🎯 Progressive Question Ladders
- **Prior Ladders**: Backward reasoning that identifies foundational questions to make your current question trivial
- **Future Ladders**: Forward reasoning organized into scenario lanes with natural decision endpoints

### 🧠 Cognitive Progression
Questions follow a natural cognitive progression:
```
DEFINE → SCOPE → QUANTIFY → COMPARE → SIMULATE → DECIDE → COMMIT
```

### 🌐 Ontology Management
- Extracts entities, relations, and assertions from questions and context
- Maintains knowledge graph with provenance and evidence tracking
- Links questions to ontological mentions and claims

### ⚡ Parallel Processing Pipeline
1. **ENUMERATE**: Generate diverse candidates (fast/cheap model)
2. **RERANK**: Score by expected info gain × coherence (reranker)
3. **STITCH**: Wire relationships and write rationales (strong model)

## Installation

### Prerequisites
- Python 3.10+
- Poetry (recommended) or pip

### Setup

1. **Clone and install dependencies**:
```bash
git clone <repository>
cd godmode
poetry install
# or: pip install -e .
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Required API Keys**:
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models) 
- Voyage AI API key (for reranking)

## Quick Start

### 1. Run the Server
```bash
python main.py
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 2. Try the Demo
```bash
python examples/demo.py
```

### 3. API Usage

**Initialize exploration**:
```bash
curl -X POST "http://localhost:8000/init" \
  -H "Content-Type: application/json" \
  -d '{
    "current_question": "Should I switch careers?",
    "context": "I am a software engineer considering product management..."
  }'
```

**Advance exploration**:
```bash
curl -X POST "http://localhost:8000/advance" \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "Q001",
    "user_answer": "Success means impact and growth"
  }'
```

## API Commands

### Core Commands
- `INIT`: Initialize new question exploration
- `ADVANCE`: Expand around a chosen node  
- `CONTINUE`: Continue deepest promising thread
- `SUMMARIZE`: Get path summary + next step recommendation
- `REGRAFT`: Move sub-branch to different lane
- `MERGE`: Merge concurrent branches

### Response Structure
```json
{
  "chat_reply": "Concise tactical response referencing key nodes",
  "graph_update": {
    "current_question": "...",
    "priors": [...],
    "scenarios": [...],
    "threads": [...],
    "meta": {...}
  },
  "ontology_update": {
    "entities": [...],
    "relations": [...], 
    "mappings": [...]
  }
}
```

## Architecture

### Core Components

- **`GodmodeEngine`**: Main orchestrator with dual-mode operation
- **`LadderGenerator`**: Progressive question ladder generation
- **`ModelRouter`**: Handles enumerate/rerank/stitch pipeline
- **`OntologyManager`**: Knowledge graph management
- **`InvariantValidator`**: Ensures response consistency

### Key Invariants

- **DAG Structure**: No cycles in question dependencies
- **Progressive Levels**: Level increases along builds_on chains  
- **Cognitive Progression**: Natural move sequences within ladders
- **Delta Nuances**: Each node adds meaningful new dimension
- **No Duplicates**: Unique questions at same level within lanes

## Configuration

### Model Configuration
```python
model_config = ModelConfig(
    enumerate_model="gpt-3.5-turbo",    # Fast/cheap for generation
    stitch_model="gpt-4-turbo-preview", # Strong for reasoning
    rerank_model="voyage-rerank-2.5"    # Reranking service
)
```

### Ladder Configuration  
```python
ladder_config = LadderConfig(
    beam_width=4,        # Candidates per layer
    depth_back=4,        # Prior ladder depth
    depth_forward=5,     # Future ladder depth
    prune_threshold=0.18 # Info gain threshold
)
```

## Examples

### Career Decision
```python
question = "Should I switch from engineering to product management?"
context = "6 years experience, enjoy strategy, less coding fulfillment..."

# GODMODE generates:
# Priors: Define success → Scope constraints → Quantify trade-offs
# Futures: Career-Max lane, Lifestyle-Opt lane, Option-Value lane
```

### Business Strategy
```python
question = "Should we expand to international markets?"
context = "SaaS company, $10M ARR, strong US presence..."

# GODMODE generates:
# Priors: Market definition → Resource requirements → Risk assessment  
# Futures: Aggressive-Growth, Conservative-Test, Partnership lanes
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black godmode/
isort godmode/
mypy godmode/
```

### Project Structure
```
godmode/
├── __init__.py          # Package entry point
├── engine.py            # Core GODMODE engine
├── schemas.py           # Pydantic schemas
├── models.py            # Model routing & interfaces
├── ladder_generator.py  # Question ladder generation
├── ontology.py          # Knowledge graph management
├── validator.py         # Invariant validation
└── api.py              # FastAPI interface
```

## Performance

- **Latency**: ~2-3s per exploration (configurable)
- **Parallel Processing**: Enumerate/rerank/stitch pipeline
- **Caching**: Question canonicalization for deduplication
- **Budget Awareness**: Configurable time/token limits

## Roadmap

- [ ] Neo4j integration for persistent knowledge graph
- [ ] Advanced cross-lane junction detection
- [ ] Learned scoring models vs heuristics  
- [ ] Real-time collaborative exploration
- [ ] Mobile-optimized tree UI components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[License details to be added]

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Run the health check at `/health`

---

**GODMODE**: *See the questions before you ask them.* 🧠✨