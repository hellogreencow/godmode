# 🧠 GODMODE - Superhuman Question Foresight Engine

> **"See the questions before you ask them."**

GODMODE is a revolutionary AI-powered system that transforms how we approach complex decisions by generating comprehensive question ladders before you even know what to ask.

## 🎯 What is GODMODE?

GODMODE operates as a **Question Foresight Engine** with dual capabilities:

- **🔙 Backward Reasoning (PRIOR Ladders)**: Discovers the foundational questions you should have asked first
- **🔜 Forward Reasoning (FUTURE Scenarios)**: Explores multiple decision paths and their implications
- **🧠 Ontological Grounding**: Builds knowledge graphs connecting concepts, entities, and relationships
- **⚡ Parallel Processing**: Uses multiple AI models in concert for speed and accuracy

## 🚀 Key Features

### Core Capabilities
- **Progressive Question Ladders**: Each question builds logically on previous ones
- **Cognitive Move Progression**: Follows structured thinking patterns (define → scope → quantify → compare → simulate → decide → commit)
- **Multi-Scenario Planning**: Generates 3-5 alternative approach lanes
- **Real-time Validation**: Ensures logical consistency and prevents circular reasoning
- **Knowledge Graph Integration**: Extracts and links entities, relations, and claims

### Architecture Highlights
- **Three-Phase Pipeline**: ENUMERATE → RERANK → STITCH for optimal quality
- **Budget Management**: Respects time, token, and computational constraints
- **Memory Architecture**: Short-term, long-term, and recall systems for context
- **Invariant Validation**: Maintains DAG structure and logical progression

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-org/godmode.git
cd godmode

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🎮 Quick Start

### Command Line Interface

```bash
# Analyze a business question
godmode init "How should we improve customer satisfaction?"

# Interactive session
godmode interactive

# Run demo with sample questions
godmode demo
```

### Python API

```python
import asyncio
from godmode import GodmodeEngine
from godmode.models.commands import InitCommand

async def main():
    engine = GodmodeEngine()
    
    command = InitCommand(
        current_question="Should we expand to the European market?",
        context="B2B SaaS company, $5M ARR, 200 customers in North America"
    )
    
    response = await engine.process_command(command)
    
    print(f"💬 {response.chat_reply}")
    print(f"📋 Generated {len(response.graph_update.priors)} prior questions")
    print(f"🚀 Created {len(response.graph_update.scenarios)} scenario lanes")

asyncio.run(main())
```

## 📊 Example Output

For the question: **"How should we improve customer satisfaction?"**

### 🔙 Prior Questions (Backward Reasoning)
```
QP1: What criteria define 'optimal' or 'best' in this context?
  ├─ Cognitive Move: define
  ├─ Info Gain: 0.90
  └─ Builds On: None

QP2: What is the current baseline we're improving from?
  ├─ Cognitive Move: scope  
  ├─ Info Gain: 0.75
  └─ Builds On: ['QP1']

QP3: What methods and approaches are available to us?
  ├─ Cognitive Move: quantify
  ├─ Info Gain: 0.60
  └─ Builds On: ['QP2']
```

### 🚀 Future Scenarios (Forward Reasoning)
```
S-A: Direct Path (Efficiency-focused)
├─ QA1: How do we define success for this efficiency approach?
├─ QA2: What are the boundaries and constraints?
├─ QA3: How do we measure progress?
└─ QA4: How does this compare to alternatives?

S-B: Comprehensive (Thoroughness-focused)
├─ QB1: How do we define success for comprehensive exploration?
├─ QB2: What are all possible approaches?
└─ QB3: What are the trade-offs between approaches?

S-C: Risk-Managed (Safety-focused)
├─ QC1: What are the potential risks and downsides?
├─ QC2: How do we mitigate major risks?
└─ QC3: What's our fallback strategy?
```

## 🏗️ Architecture

### Core Components

```
🧠 GODMODE Engine
├── 🔙 Backward Reasoning
│   ├── Premise Extraction
│   ├── Question Generation  
│   └── Ladder Construction
├── 🔜 Forward Reasoning
│   ├── Scenario Generation
│   ├── Lane Development
│   └── Natural Endings
├── 🎯 Ontology Manager
│   ├── Entity Extraction
│   ├── Relation Mapping
│   └── Knowledge Graph
├── 🧮 Validation System
│   ├── DAG Verification
│   ├── Level Progression
│   └── Invariant Checking
└── 💾 Memory System
    ├── Short-term Context
    ├── Long-term Patterns
    └── Recall Agent
```

### Processing Pipeline

```
1. ENUMERATE 🔄
   ├── Extract premises from question
   ├── Generate diverse candidates
   └── Create scenario seeds

2. RERANK 📊  
   ├── Score by info_gain × coherence
   ├── Apply effort penalties
   └── Sort by expected value

3. STITCH 🔗
   ├── Wire builds_on relationships
   ├── Add triggers and cross-links
   └── Validate invariants
```

## 🎯 Use Cases

### Business Strategy
- **Market Entry**: "Should we expand to Asia?"
- **Product Development**: "What features should we build next?"
- **Team Scaling**: "How do we grow from 10 to 100 engineers?"

### Personal Decisions
- **Career Transitions**: "Should I switch from engineering to product management?"
- **Investment Choices**: "Is now the right time to buy a house?"
- **Education Planning**: "Should I pursue an MBA or gain more experience?"

### Research & Analysis
- **Problem Decomposition**: Break complex problems into manageable questions
- **Scenario Planning**: Explore multiple future paths systematically
- **Risk Assessment**: Identify hidden assumptions and potential pitfalls

## 🧪 Testing

```bash
# Run basic functionality test
python3 test_godmode_basic.py

# Run comprehensive demo
python3 demo_godmode.py

# Unit tests (requires pytest)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

## 📈 Performance

GODMODE is designed for speed and efficiency:

- **Parallel Processing**: Multiple AI models work simultaneously
- **Budget Management**: Respects time and computational limits
- **Smart Caching**: Avoids redundant computations
- **Progressive Depth**: Starts broad, deepens selectively

Typical performance on modern hardware:
- **Simple Questions**: < 2 seconds
- **Complex Analysis**: < 5 seconds  
- **Deep Exploration**: < 10 seconds

## 🔧 Configuration

### Budget Settings
```python
from godmode.models.commands import Budgets

budgets = Budgets(
    beam_width=4,           # Candidates per layer
    depth_back=4,           # Prior ladder depth
    depth_fwd=5,            # Future scenario depth
    max_tokens_reply=160,   # Chat response limit
    time_s=2.5,             # Processing time limit
    prune_if_info_gain_below=0.18  # Quality threshold
)
```

### Model Configuration
```python
# Configure AI model routing
config = {
    "enumerate_model": "gpt-3.5-turbo",    # Fast candidate generation
    "rerank_model": "voyage-reranker-2",   # Precise ranking
    "stitch_model": "gpt-4",               # Complex reasoning
    "fact_check_model": "claude-3-sonnet"  # Verification
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black godmode/
isort godmode/

# Type checking
mypy godmode/
```

## 📚 Documentation

- **[API Reference](docs/api.md)**: Complete API documentation
- **[Architecture Guide](docs/architecture.md)**: Deep dive into system design
- **[Examples](docs/examples.md)**: Comprehensive usage examples
- **[Contributing](CONTRIBUTING.md)**: How to contribute to GODMODE

## 🎭 Philosophy

GODMODE is built on the principle that **the quality of your decisions depends on the quality of your questions**. By systematically exploring the question space before diving into answers, we can:

1. **Avoid Blind Spots**: Discover assumptions we didn't know we had
2. **Explore Alternatives**: Consider paths we might have missed
3. **Make Better Decisions**: Ground choices in comprehensive analysis
4. **Learn Continuously**: Build knowledge graphs that grow over time

## 🏆 Why GODMODE?

Traditional decision-making tools focus on **answers**. GODMODE focuses on **questions**.

| Traditional Approach | GODMODE Approach |
|---------------------|------------------|
| Start with solutions | Start with premises |
| Single path analysis | Multiple scenario lanes |
| Static frameworks | Dynamic question generation |
| Isolated decisions | Connected knowledge graphs |
| Reactive thinking | Proactive foresight |

## 🚧 Roadmap

- [ ] **Web Interface**: Interactive dual-rail tree visualization
- [ ] **Real-time Collaboration**: Multi-user question exploration
- [ ] **Integration APIs**: Connect with existing tools and workflows  
- [ ] **Advanced Models**: Support for specialized domain models
- [ ] **Enterprise Features**: SSO, audit logs, compliance tools

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

GODMODE draws inspiration from:
- **Socratic Questioning**: The power of asking the right questions
- **Systems Thinking**: Understanding interconnections and feedback loops
- **Decision Theory**: Structured approaches to complex choices
- **Knowledge Graphs**: Representing and reasoning with connected information

---

**Ready to see the questions before you ask them?**

🧠 **[Try GODMODE Now](https://github.com/your-org/godmode)** 🧠