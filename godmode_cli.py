#!/usr/bin/env python3
"""
Simple GodMode CLI - Easy commands for hierarchical reasoning
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import GodMode components
try:
    from godmode import GodModeEngine, Problem
    from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel
    from godmode.integrations.openrouter import OpenRouterIntegration
    from godmode.web.app import GodModeWebApp
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -e .")
    sys.exit(1)

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def godmode():
    """🧠 GodMode: Advanced Hierarchical Reasoning System"""
    pass

@godmode.command()
@click.option('--port', default=8000, help='Port for web server')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--gpu/--no-gpu', default=True, help='Enable GPU acceleration')
def start(port, host, gpu):
    """🚀 Start the GodMode web interface"""
    console.print(Panel(
        f"[bold green]Starting GodMode Web Interface[/bold green]\n\n"
        f"🌐 URL: http://{host}:{port}\n"
        f"🧪 Experiment: http://localhost:{port}/experiment\n"
        f"📊 Demo: http://localhost:{port}/demo\n"
        f"📖 API: http://localhost:{port}/api/docs",
        title="🧠 GodMode Server"
    ))
    
    try:
        with console.status("[bold blue]Initializing systems..."):
            engine = GodModeEngine(enable_gpu=gpu)
            hrm_model = HierarchicalReasoningModel(use_pretrained=True)
            web_app = GodModeWebApp(engine=engine, hierarchical_model=hrm_model)
        
        console.print("✅ Systems initialized!")
        console.print(f"🚀 Starting server on {host}:{port}")
        
        web_app.run(host=host, port=port)
        
    except KeyboardInterrupt:
        console.print("\n👋 Server stopped")
    except Exception as e:
        console.print(f"❌ Error: {e}")

@godmode.command()
@click.argument('problem_text')
@click.option('--model', help='Specific model to use (e.g., claude-3.5-sonnet)')
@click.option('--local/--api', default=False, help='Force local or API processing')
@click.option('--save', help='Save result to file')
def solve(problem_text, model, local, save):
    """🎯 Solve a problem with hierarchical reasoning"""
    console.print(Panel(f"[bold blue]Problem:[/bold blue] {problem_text}", title="🧠 Reasoning Task"))
    
    async def solve_async():
        problem = Problem(
            title="CLI Problem",
            description=problem_text,
            problem_type="general",
            domain="general"
        )
        
        if local or not os.getenv('OPENROUTER_API_KEY'):
            # Use local HRM
            console.print("🧠 Using local Hierarchical Reasoning Model...")
            with console.status("[bold yellow]Processing with HRM..."):
                hrm = HierarchicalReasoningModel(embedding_dim=256, use_pretrained=False)
                result = hrm.solve_problem(problem)
            
            if result['solutions']:
                solution = result['solutions'][0]
                console.print(f"✅ [bold green]Solution found![/bold green]")
                console.print(f"🎯 Confidence: {result['confidence']:.1%}")
                console.print(f"🔄 Levels used: {len(result['reasoning_trace']['levels_activated'])}")
                console.print(Panel(solution.solution_text, title="💡 HRM Solution"))
                
                return {
                    "solution": solution.solution_text,
                    "confidence": result['confidence'],
                    "method": "local_hrm",
                    "levels": result['reasoning_trace']['levels_activated']
                }
        else:
            # Use API
            console.print("🌐 Using external API model...")
            openrouter = OpenRouterIntegration()
            
            try:
                model_id = f"anthropic/{model}" if model and not "/" in model else (model or "anthropic/claude-3.5-sonnet")
                
                with console.status(f"[bold yellow]Processing with {model_id}..."):
                    solution = await openrouter.solve_problem_with_model(problem, model_id)
                
                console.print(f"✅ [bold green]Solution found![/bold green]")
                console.print(f"🎯 Confidence: {solution.confidence:.1%}")
                console.print(f"🤖 Model: {model_id}")
                console.print(Panel(solution.solution_text, title="💡 API Solution"))
                
                return {
                    "solution": solution.solution_text,
                    "confidence": solution.confidence,
                    "method": "api",
                    "model": model_id
                }
                
            finally:
                await openrouter.close()
    
    try:
        result = asyncio.run(solve_async())
        
        if save:
            with open(save, 'w') as f:
                json.dump(result, f, indent=2)
            console.print(f"💾 Result saved to {save}")
            
    except Exception as e:
        console.print(f"❌ Error: {e}")

@godmode.command()
def demo():
    """🎮 Run interactive demonstration"""
    console.print(Panel(
        "[bold magenta]🎮 Interactive GodMode Demo[/bold magenta]\n\n"
        "This will demonstrate:\n"
        "• Hierarchical reasoning across 4 cognitive levels\n"
        "• Cross-level attention mechanisms\n" 
        "• Local vs API model comparison\n"
        "• Real-time reasoning visualization",
        title="Demo Mode"
    ))
    
    try:
        from demo_godmode import main as demo_main
        asyncio.run(demo_main())
    except Exception as e:
        console.print(f"❌ Demo error: {e}")

@godmode.command()
def test():
    """🧪 Run system tests"""
    console.print(Panel(
        "[bold blue]🧪 Running GodMode Tests[/bold blue]\n\n"
        "Testing:\n"
        "• Core engine functionality\n"
        "• Hierarchical reasoning model\n"
        "• Memory systems\n"
        "• API integrations",
        title="Test Suite"
    ))
    
    try:
        from test_godmode_basic import run_all_tests
        success = asyncio.run(run_all_tests())
        if success:
            console.print("🎉 All tests passed!")
        else:
            console.print("⚠️ Some tests failed")
    except Exception as e:
        console.print(f"❌ Test error: {e}")

@godmode.command()
@click.option('--show-models', is_flag=True, help='Show available models')
def models(show_models):
    """🤖 Manage AI models"""
    if show_models:
        async def show_models_async():
            if not os.getenv('OPENROUTER_API_KEY'):
                console.print("❌ Set OPENROUTER_API_KEY to view external models")
                return
            
            openrouter = OpenRouterIntegration()
            try:
                with console.status("[bold blue]Fetching available models..."):
                    models = await openrouter.get_available_models()
                
                table = Table(title="🤖 Available Models")
                table.add_column("Model", style="cyan")
                table.add_column("Provider", style="green") 
                table.add_column("Context", style="yellow")
                table.add_column("Best For", style="magenta")
                
                top_models = [
                    ("anthropic/claude-3.5-sonnet", "Anthropic", "200K", "Complex reasoning"),
                    ("openai/gpt-4-turbo-preview", "OpenAI", "128K", "General purpose"),
                    ("google/gemini-pro-1.5", "Google", "1M", "Large context"),
                    ("meta-llama/llama-3.1-405b-instruct", "Meta", "128K", "Open source"),
                ]
                
                for model_id, provider, context, best_for in top_models:
                    model_info = next((m for m in models if m.id == model_id), None)
                    if model_info:
                        table.add_row(model_info.name, provider, context, best_for)
                
                console.print(table)
                console.print(f"\n[dim]Total models available: {len(models)}[/dim]")
                
            finally:
                await openrouter.close()
        
        asyncio.run(show_models_async())
    else:
        # Show local model info
        console.print(Panel(
            "[bold blue]Local Hierarchical Reasoning Model[/bold blue]\n\n"
            "🧠 Architecture: 4-level cognitive hierarchy\n"
            "⚡ Attention: Multi-level cross-attention\n"
            "🔧 Size: ~1GB (including embeddings)\n"
            "💾 RAM: 8GB minimum, 16GB recommended\n"
            "🚀 GPU: Optional but recommended\n\n"
            "[yellow]Cognitive Levels:[/yellow]\n"
            "• Metacognitive: Strategic planning\n"
            "• Executive: Goal management\n" 
            "• Operational: Task execution\n"
            "• Reactive: Quick responses",
            title="🧠 HRM Info"
        ))

@godmode.command()
def config():
    """⚙️ Show configuration and setup status"""
    table = Table(title="⚙️ GodMode Configuration")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = sys.version_info >= (3, 11)
    table.add_row(
        "Python Version", 
        "✅ OK" if python_ok else "❌ Too Old",
        f"{python_version} ({'✅' if python_ok else '❌ Need 3.11+'})"
    )
    
    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        table.add_row(
            "GPU Support",
            "✅ Available" if gpu_available else "⚠️ CPU Only", 
            f"{gpu_count} GPU(s)" if gpu_available else "No CUDA devices"
        )
    except ImportError:
        table.add_row("GPU Support", "❌ PyTorch Missing", "Run: pip install torch")
    
    # Check API Key
    api_key = os.getenv('OPENROUTER_API_KEY')
    table.add_row(
        "OpenRouter API",
        "✅ Configured" if api_key else "⚠️ Not Set",
        "External models available" if api_key else "Set OPENROUTER_API_KEY for external models"
    )
    
    # Check dependencies
    try:
        import godmode
        table.add_row("GodMode Core", "✅ Installed", "All core components available")
    except ImportError:
        table.add_row("GodMode Core", "❌ Missing", "Run: pip install -e .")
    
    console.print(table)
    
    # Show quick setup if issues found
    if not python_ok or not api_key:
        console.print(Panel(
            "[bold yellow]Quick Setup:[/bold yellow]\n\n"
            "1. Get OpenRouter API key: https://openrouter.ai/\n"
            "2. Set environment variable: export OPENROUTER_API_KEY='your-key'\n"
            "3. Install dependencies: pip install -e .\n"
            "4. Test system: godmode test\n"
            "5. Start web interface: godmode start",
            title="🔧 Setup Guide"
        ))

@godmode.command()
def benchmark():
    """📊 Run performance benchmarks"""
    console.print(Panel(
        "[bold blue]📊 Running Performance Benchmarks[/bold blue]\n\n"
        "Testing:\n"
        "• Local HRM performance\n"
        "• API model comparison\n"
        "• Memory usage analysis\n"
        "• Response time metrics",
        title="Benchmark Suite"
    ))
    
    # Simple benchmark
    problems = [
        "Design a sustainable transportation system",
        "Optimize supply chain efficiency", 
        "Create an AI ethics framework"
    ]
    
    async def run_benchmark():
        results = []
        
        for problem_text in problems:
            problem = Problem(
                title=f"Benchmark: {problem_text[:30]}...",
                description=problem_text,
                problem_type="general",
                domain="general"
            )
            
            # Test local HRM
            import time
            start_time = time.time()
            hrm = HierarchicalReasoningModel(embedding_dim=128, use_pretrained=False)
            result = hrm.solve_problem(problem)
            local_time = time.time() - start_time
            
            results.append({
                "problem": problem_text[:30] + "...",
                "local_time": local_time,
                "local_confidence": result['confidence'],
                "local_solutions": len(result['solutions'])
            })
        
        # Display results
        table = Table(title="📊 Benchmark Results")
        table.add_column("Problem", style="cyan")
        table.add_column("Time (s)", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Solutions", style="blue")
        
        for r in results:
            table.add_row(
                r["problem"],
                f"{r['local_time']:.2f}",
                f"{r['local_confidence']:.1%}",
                str(r["local_solutions"])
            )
        
        console.print(table)
        
        avg_time = sum(r["local_time"] for r in results) / len(results)
        avg_conf = sum(r["local_confidence"] for r in results) / len(results)
        
        console.print(f"\n📈 Average Performance:")
        console.print(f"⏱️  Response Time: {avg_time:.2f}s")
        console.print(f"🎯 Confidence: {avg_conf:.1%}")
    
    try:
        asyncio.run(run_benchmark())
    except Exception as e:
        console.print(f"❌ Benchmark error: {e}")

if __name__ == '__main__':
    godmode()