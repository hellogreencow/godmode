#!/usr/bin/env python3
"""
GodMode demonstration script showcasing hierarchical reasoning capabilities.

This script demonstrates the advanced features of the GodMode system including
hierarchical reasoning models, multi-level cognitive architectures, and
real-time problem solving.
"""

import asyncio
import time
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich import print as rprint

from godmode import (
    GodModeEngine,
    HierarchicalReasoningModel,
    Problem,
    ReasoningType,
    CognitiveLevel,
)


console = Console()


def create_demo_problems():
    """Create a set of demonstration problems."""
    problems = [
        Problem(
            title="Sustainable City Transportation",
            description="Design a transportation system for a city of 2 million people that reduces carbon emissions by 60% while maintaining accessibility and economic viability",
            problem_type="planning",
            domain="urban_planning",
            constraints=[
                "Must reduce emissions by 60%",
                "Budget limit of $50 billion",
                "Implementation within 15 years",
                "Maintain current accessibility levels"
            ],
            objectives=[
                "Reduce carbon emissions",
                "Maintain accessibility",
                "Ensure economic viability",
                "Improve quality of life"
            ]
        ),
        Problem(
            title="AI Learning Without Forgetting",
            description="Develop an AI system that can continuously learn new tasks and domains without forgetting previously learned knowledge",
            problem_type="design",
            domain="artificial_intelligence",
            constraints=[
                "No catastrophic forgetting",
                "Real-time learning capability",
                "Scalable to 1000+ domains",
                "Memory efficient"
            ],
            objectives=[
                "Continuous learning",
                "Knowledge retention",
                "Scalability",
                "Efficiency"
            ]
        ),
        Problem(
            title="Global Supply Chain Optimization",
            description="Optimize a global supply chain network to minimize costs while maximizing resilience to disruptions and reducing environmental impact",
            problem_type="optimization",
            domain="logistics",
            constraints=[
                "Cost reduction of 25%",
                "Resilience to 95% of disruptions",
                "30% reduction in environmental impact",
                "Maintain service levels"
            ],
            objectives=[
                "Minimize total costs",
                "Maximize resilience",
                "Reduce environmental impact",
                "Maintain service quality"
            ]
        )
    ]
    
    return problems


async def demonstrate_standard_reasoning(engine: GodModeEngine, problem: Problem):
    """Demonstrate standard reasoning engine capabilities."""
    console.print(f"\n[bold blue]Standard Reasoning Engine[/bold blue]")
    console.print(f"Problem: {problem.title}")
    
    start_time = time.time()
    
    with console.status("[bold green]Reasoning in progress..."):
        response = await engine.solve_problem(
            problem=problem,
            reasoning_type=ReasoningType.HIERARCHICAL,
            max_time=30.0,
            min_confidence=0.6
        )
    
    solving_time = time.time() - start_time
    
    # Display results
    table = Table(title="Standard Engine Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Solving Time", f"{solving_time:.2f}s")
    table.add_row("Solution Found", "✅" if response.solution else "❌")
    table.add_row("Confidence", f"{response.solution.confidence:.1%}" if response.solution else "N/A")
    table.add_row("Reasoning Steps", str(response.total_steps))
    table.add_row("Success Rate", f"{response.success_rate:.1%}")
    
    if response.solution:
        table.add_row("Quality Score", f"{response.solution.get_overall_quality():.1%}")
    
    console.print(table)
    
    if response.solution:
        console.print(Panel(
            response.solution.solution_text[:300] + "..." if len(response.solution.solution_text) > 300 else response.solution.solution_text,
            title="Solution Preview"
        ))
    
    return response


def demonstrate_hierarchical_model(model: HierarchicalReasoningModel, problem: Problem):
    """Demonstrate hierarchical reasoning model capabilities."""
    console.print(f"\n[bold magenta]Hierarchical Reasoning Model[/bold magenta]")
    console.print(f"Problem: {problem.title}")
    
    start_time = time.time()
    
    with console.status("[bold yellow]Hierarchical reasoning in progress..."):
        result = model.solve_problem(problem)
    
    solving_time = time.time() - start_time
    
    # Display results
    table = Table(title="Hierarchical Model Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Solving Time", f"{solving_time:.2f}s")
    table.add_row("Solutions Generated", str(len(result["solutions"])))
    table.add_row("Overall Confidence", f"{result['confidence']:.1%}")
    table.add_row("Levels Activated", str(len(result["reasoning_trace"]["levels_activated"])))
    table.add_row("Processing Steps", str(result["reasoning_trace"]["processing_steps"]))
    
    console.print(table)
    
    if result["solutions"]:
        best_solution = result["solutions"][0]
        console.print(Panel(
            best_solution.solution_text[:300] + "..." if len(best_solution.solution_text) > 300 else best_solution.solution_text,
            title="Best Solution Preview"
        ))
    
    return result


def compare_approaches(standard_result, hierarchical_result):
    """Compare results from both approaches."""
    console.print(f"\n[bold red]Approach Comparison[/bold red]")
    
    comparison_table = Table(title="Performance Comparison")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Standard Engine", style="blue")
    comparison_table.add_column("Hierarchical Model", style="magenta")
    comparison_table.add_column("Winner", style="green")
    
    # Compare confidence
    std_conf = standard_result.solution.confidence if standard_result.solution else 0
    hier_conf = hierarchical_result["confidence"]
    
    comparison_table.add_row(
        "Confidence",
        f"{std_conf:.1%}",
        f"{hier_conf:.1%}",
        "Hierarchical" if hier_conf > std_conf else "Standard"
    )
    
    # Compare solution count
    std_solutions = 1 if standard_result.solution else 0
    hier_solutions = len(hierarchical_result["solutions"])
    
    comparison_table.add_row(
        "Solutions Generated",
        str(std_solutions),
        str(hier_solutions),
        "Hierarchical" if hier_solutions > std_solutions else "Standard"
    )
    
    # Compare reasoning depth
    std_steps = standard_result.total_steps
    hier_steps = hierarchical_result["reasoning_trace"]["processing_steps"]
    
    comparison_table.add_row(
        "Reasoning Depth",
        str(std_steps),
        str(hier_steps),
        "Hierarchical" if hier_steps > std_steps else "Standard"
    )
    
    console.print(comparison_table)


async def demonstrate_memory_system(engine: GodModeEngine):
    """Demonstrate the cognitive memory system."""
    console.print(f"\n[bold cyan]Cognitive Memory System[/bold cyan]")
    
    # Store some sample knowledge
    sample_knowledge = [
        {"text": "Transportation systems should prioritize sustainability", "type": "principle"},
        {"text": "AI systems benefit from hierarchical architectures", "type": "insight"},
        {"text": "Supply chains need redundancy for resilience", "type": "best_practice"},
        {"text": "Urban planning requires stakeholder engagement", "type": "methodology"},
    ]
    
    console.print("Storing knowledge in memory...")
    for knowledge in sample_knowledge:
        await engine.memory.store(knowledge, importance=0.8)
    
    # Demonstrate retrieval
    console.print("\nTesting memory retrieval:")
    
    queries = [
        "transportation sustainability",
        "AI architecture",
        "supply chain resilience"
    ]
    
    for query in queries:
        results = await engine.memory.retrieve(query, top_k=2)
        console.print(f"\nQuery: '{query}'")
        for i, result in enumerate(results, 1):
            console.print(f"  {i}. {result.content.get('text', 'N/A')} (activation: {result.get_activation():.2f})")
    
    # Show memory statistics
    stats = engine.memory.get_statistics()
    
    memory_table = Table(title="Memory Statistics")
    memory_table.add_column("Metric", style="cyan")
    memory_table.add_column("Value", style="green")
    
    for key, value in stats.items():
        memory_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(memory_table)


def demonstrate_cognitive_levels():
    """Demonstrate the different cognitive levels."""
    console.print(f"\n[bold yellow]Cognitive Architecture Levels[/bold yellow]")
    
    levels_table = Table(title="Hierarchical Cognitive Levels")
    levels_table.add_column("Level", style="cyan")
    levels_table.add_column("Function", style="green")
    levels_table.add_column("Example Operations", style="yellow")
    
    level_info = {
        CognitiveLevel.METACOGNITIVE: {
            "function": "Strategic planning and meta-reasoning",
            "operations": "Problem decomposition, strategy selection, resource allocation"
        },
        CognitiveLevel.EXECUTIVE: {
            "function": "Goal management and control",
            "operations": "Goal prioritization, execution planning, monitoring"
        },
        CognitiveLevel.OPERATIONAL: {
            "function": "Task execution and procedures",
            "operations": "Algorithm application, data processing, rule execution"
        },
        CognitiveLevel.REACTIVE: {
            "function": "Immediate responses and reflexes",
            "operations": "Pattern matching, heuristic application, quick responses"
        }
    }
    
    for level, info in level_info.items():
        levels_table.add_row(
            level.value.title(),
            info["function"],
            info["operations"]
        )
    
    console.print(levels_table)


async def run_performance_benchmark(engine: GodModeEngine, model: HierarchicalReasoningModel, problems):
    """Run a performance benchmark on multiple problems."""
    console.print(f"\n[bold red]Performance Benchmark[/bold red]")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmark...", total=len(problems) * 2)
        
        for problem in problems:
            # Test standard engine
            start_time = time.time()
            std_result = await engine.solve_problem(problem, max_time=15.0)
            std_time = time.time() - start_time
            progress.advance(task)
            
            # Test hierarchical model
            start_time = time.time()
            hier_result = model.solve_problem(problem)
            hier_time = time.time() - start_time
            progress.advance(task)
            
            results.append({
                "problem": problem.title[:30] + "...",
                "std_time": std_time,
                "std_confidence": std_result.solution.confidence if std_result.solution else 0,
                "std_success": std_result.solution is not None,
                "hier_time": hier_time,
                "hier_confidence": hier_result["confidence"],
                "hier_success": len(hier_result["solutions"]) > 0,
            })
    
    # Display benchmark results
    benchmark_table = Table(title="Benchmark Results")
    benchmark_table.add_column("Problem", style="cyan")
    benchmark_table.add_column("Standard Time", style="blue")
    benchmark_table.add_column("Standard Conf", style="blue")
    benchmark_table.add_column("Hierarchical Time", style="magenta")
    benchmark_table.add_column("Hierarchical Conf", style="magenta")
    benchmark_table.add_column("Winner", style="green")
    
    for result in results:
        winner = "Hierarchical" if result["hier_confidence"] > result["std_confidence"] else "Standard"
        
        benchmark_table.add_row(
            result["problem"],
            f"{result['std_time']:.2f}s",
            f"{result['std_confidence']:.1%}",
            f"{result['hier_time']:.2f}s",
            f"{result['hier_confidence']:.1%}",
            winner
        )
    
    console.print(benchmark_table)
    
    # Summary statistics
    avg_std_time = sum(r["std_time"] for r in results) / len(results)
    avg_hier_time = sum(r["hier_time"] for r in results) / len(results)
    avg_std_conf = sum(r["std_confidence"] for r in results) / len(results)
    avg_hier_conf = sum(r["hier_confidence"] for r in results) / len(results)
    
    summary_table = Table(title="Benchmark Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Standard Engine", style="blue")
    summary_table.add_column("Hierarchical Model", style="magenta")
    
    summary_table.add_row("Average Time", f"{avg_std_time:.2f}s", f"{avg_hier_time:.2f}s")
    summary_table.add_row("Average Confidence", f"{avg_std_conf:.1%}", f"{avg_hier_conf:.1%}")
    summary_table.add_row("Success Rate", 
                         f"{sum(1 for r in results if r['std_success']) / len(results):.1%}",
                         f"{sum(1 for r in results if r['hier_success']) / len(results):.1%}")
    
    console.print(summary_table)


async def main():
    """Main demonstration function."""
    console.print(Panel(
        "[bold green]🧠 GodMode: Advanced Hierarchical Reasoning System[/bold green]\n"
        "[yellow]Demonstration of cutting-edge AI reasoning capabilities[/yellow]",
        title="Welcome to GodMode Demo"
    ))
    
    # Initialize systems
    console.print("\n[bold blue]Initializing Systems...[/bold blue]")
    
    with console.status("[bold green]Loading GodMode Engine..."):
        engine = GodModeEngine(
            memory_size=5000,
            reasoning_depth=4,
            enable_gpu=False,  # Set to True if you have CUDA
        )
    
    with console.status("[bold green]Loading Hierarchical Model..."):
        model = HierarchicalReasoningModel(
            embedding_dim=256,  # Smaller for demo
            use_pretrained=False,  # Faster initialization
        )
    
    console.print("✅ Systems initialized successfully!")
    
    # Create demo problems
    problems = create_demo_problems()
    
    # Demonstrate cognitive levels
    demonstrate_cognitive_levels()
    
    # Demonstrate memory system
    await demonstrate_memory_system(engine)
    
    # Demonstrate reasoning on first problem
    problem = problems[0]
    
    console.print(f"\n[bold green]Problem Analysis Demonstration[/bold green]")
    console.print(Panel(
        f"[bold]{problem.title}[/bold]\n\n"
        f"{problem.description}\n\n"
        f"[yellow]Constraints:[/yellow] {', '.join(problem.constraints[:2])}...\n"
        f"[yellow]Objectives:[/yellow] {', '.join(problem.objectives[:2])}...",
        title="Selected Problem"
    ))
    
    # Run both approaches
    standard_result = await demonstrate_standard_reasoning(engine, problem)
    hierarchical_result = demonstrate_hierarchical_model(model, problem)
    
    # Compare approaches
    compare_approaches(standard_result, hierarchical_result)
    
    # Run performance benchmark
    await run_performance_benchmark(engine, model, problems)
    
    # Show final statistics
    engine_stats = engine.get_statistics()
    
    final_table = Table(title="Final System Statistics")
    final_table.add_column("Metric", style="cyan")
    final_table.add_column("Value", style="green")
    
    for key, value in engine_stats.items():
        if isinstance(value, float):
            value = f"{value:.3f}"
        final_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(final_table)
    
    console.print(Panel(
        "[bold green]🎉 Demonstration Complete![/bold green]\n\n"
        "[yellow]Key Features Demonstrated:[/yellow]\n"
        "• Hierarchical reasoning across multiple cognitive levels\n"
        "• Advanced memory system with semantic retrieval\n"
        "• Multiple reasoning strategies (forward, backward, cognitive moves)\n"
        "• Real-time problem solving with confidence estimation\n"
        "• Performance comparison and benchmarking\n\n"
        "[blue]To explore more features, try:[/blue]\n"
        "• Web interface: python -m godmode.cli web\n"
        "• Interactive mode: python -m godmode.cli interactive\n"
        "• Custom problems: python -m godmode.cli solve \"your problem here\"",
        title="Demo Summary"
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Demo interrupted. Thank you for trying GodMode!")
    except Exception as e:
        console.print(f"\n❌ Error: {e}")
        console.print("Please check your installation and try again.")
