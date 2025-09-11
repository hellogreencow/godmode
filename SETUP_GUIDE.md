# 🚀 GodMode Setup Guide

## Quick Start (Local System Only)

The original GodMode system works entirely locally without any API keys:

```bash
# 1. Switch to correct branch
git checkout cursor/advanced-project-review-and-tech-integration-b9e7

# 2. Install dependencies
pip install -e ".[dev,gpu,experimental]"

# 3. Run basic test
python test_godmode_basic.py

# 4. Try the demo
python demo_godmode.py

# 5. Launch web interface
python -m godmode.cli web
# Then visit: http://localhost:8000/experiment
```

## 🔑 Enhanced Setup with OpenRouter (Optional)

To use external models like Claude, GPT-4, Llama, etc.:

### 1. Get OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Create account and get API key
3. Add credits to your account

### 2. Set Environment Variable

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="your_key_here"
```

**Windows:**
```cmd
set OPENROUTER_API_KEY=your_key_here
```

**Or create `.env` file:**
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 3. Install Enhanced Dependencies

```bash
pip install httpx python-dotenv
```

### 4. Run Enhanced Demo

```bash
python godmode_with_openrouter.py
```

## 🤖 Available Models

### Top Reasoning Models
- **Claude 3.5 Sonnet** - Best for complex reasoning
- **GPT-4 Turbo** - Excellent all-around performance  
- **Llama 3.1 405B** - Open source powerhouse
- **Gemini Pro 1.5** - Massive context (1M tokens)

### Fast & Cost-Effective
- **Claude 3 Haiku** - Fast and affordable
- **GPT-3.5 Turbo** - Quick responses
- **Mistral 7B** - Open source, efficient

### Specialized
- **GPT-4 Vision** - Image understanding
- **Code Llama** - Programming tasks
- **Mixtral 8x7B** - Mixture of experts

## 🎮 Usage Examples

### CLI Usage
```bash
# Solve with local system
godmode solve "Design sustainable city transport"

# Solve with external model (if OpenRouter set up)
godmode solve "Design sustainable city transport" --model "anthropic/claude-3.5-sonnet"

# Interactive mode
godmode interactive

# Launch web interface
godmode web --port 8000
```

### Python API
```python
import asyncio
from godmode import GodModeEngine
from godmode.integrations.openrouter import OpenRouterIntegration

# Local reasoning
engine = GodModeEngine()
result = await engine.solve_problem("Your problem here")

# External model (if API key set)
openrouter = OpenRouterIntegration()
solution = await openrouter.solve_problem_with_model(
    problem, 
    model_id="anthropic/claude-3.5-sonnet"
)
```

### Web Interface
```bash
# Start server
python -m godmode.cli web

# Access points:
# http://localhost:8000 - Main interface
# http://localhost:8000/experiment - HRM demo  
# http://localhost:8000/demo - Interactive demo
# http://localhost:8000/api/docs - API docs
```

## 🔧 Configuration Options

### Environment Variables
```bash
# OpenRouter
OPENROUTER_API_KEY=your_key
OPENROUTER_DEFAULT_MODEL=anthropic/claude-3.5-sonnet

# System
ENABLE_GPU=true
MEMORY_SIZE=10000
REASONING_DEPTH=5

# Web
WEB_PORT=8000
DEBUG_MODE=false
```

### Model Selection
```python
from godmode.integrations.model_selector import ModelSelector

selector = ModelSelector(openrouter)
recommendations = await selector.recommend_model(
    problem,
    user_preferences={
        "budget_conscious": True,  # Prefer cost-effective models
        "speed_priority": False,   # Don't prioritize speed
    }
)
```

## 🧠 Hybrid Reasoning

Combine local HRM with external models:

```python
# Local hierarchical reasoning
from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel

hrm = HierarchicalReasoningModel()
local_result = hrm.solve_problem(problem)

# External model reasoning  
external_result = await openrouter.solve_problem_with_model(
    problem, "anthropic/claude-3.5-sonnet"
)

# Compare and combine results
best_solution = max([local_result, external_result], 
                   key=lambda x: x.confidence)
```

## 📊 Model Comparison

The system can automatically compare multiple models:

```python
models_to_test = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4-turbo-preview", 
    "meta-llama/llama-3.1-405b-instruct"
]

results = []
for model in models_to_test:
    result = await openrouter.solve_problem_with_model(problem, model)
    results.append((model, result))

# Results include confidence, quality scores, timing, etc.
```

## 🎯 Problem Types & Model Selection

The system automatically selects optimal models based on problem characteristics:

- **Complex Reasoning**: Claude 3.5 Sonnet, GPT-4 Turbo
- **Creative Tasks**: Claude 3.5 Sonnet, GPT-4 
- **Analytical Work**: GPT-4, Gemini Pro
- **Fast Responses**: Claude Haiku, GPT-3.5 Turbo
- **Cost-Effective**: Llama 3.1, Mixtral, Claude Haiku

## 🚀 Performance Tips

1. **Use GPU**: Set `ENABLE_GPU=true` for faster local processing
2. **Model Selection**: Let the system auto-select models for best results
3. **Batch Processing**: Process multiple problems together for efficiency
4. **Caching**: Results are cached to avoid redundant API calls
5. **Hybrid Approach**: Use local HRM for fast iteration, external models for final solutions

## 🔍 Troubleshooting

### Common Issues

**"No OpenRouter API key"**
- Set `OPENROUTER_API_KEY` environment variable
- Or create `.env` file with your key

**"CUDA not available"**
- Install PyTorch with CUDA support: `pip install torch[cuda]`
- Or set `ENABLE_GPU=false` to use CPU only

**"Model not found"**
- Check available models: `await openrouter.get_available_models()`
- Use model selector for recommendations

**"Import errors"**
- Install all dependencies: `pip install -e ".[dev,gpu,experimental]"`
- Check Python version (3.11+ required)

### Getting Help

1. Run diagnostics: `python test_godmode_basic.py`
2. Check logs in console output
3. Verify API key and credits on OpenRouter dashboard
4. Try local-only mode first to isolate issues

## 📈 Next Steps

1. **Start Local**: Get familiar with local HRM system first
2. **Add OpenRouter**: Enhance with external models when needed  
3. **Experiment**: Try different models for different problem types
4. **Optimize**: Use model selector and hybrid approaches
5. **Scale**: Deploy web interface for team usage

---

**🎉 You're ready to use the world's most advanced hierarchical reasoning system!**