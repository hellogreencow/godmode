#!/usr/bin/env python3
"""
Enhanced GodMode with OpenRouter integration.

This script demonstrates how to use GodMode with various external models
through OpenRouter API integration.
"""

import asyncio
import os
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from godmode import GodModeEngine, Problem
from godmode.integrations.openrouter import OpenRouterIntegration, OpenRouterConfig
from godmode.integrations.model_selector import ModelSelector


console = Console()


async def setup_openrouter() -> OpenRouterIntegration:
    """Set up OpenRouter integration."""
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        console.print(Panel(
            "[bold red]OpenRouter API Key Required[/bold red]\n\n"
            "To use external models, you need an OpenRouter API key.\n\n"
            "[yellow]Steps to get started:[/yellow]\n"
            "1. Visit https://openrouter.ai/\n"
            "2. Create an account and get your API key\n"
            "3. Set environment variable: export OPENROUTER_API_KEY='your-key-here'\n"
            "4. Or create a .env file with: OPENROUTER_API_KEY=your-key-here\n\n"
            "[cyan]Popular models available:[/cyan]\n"
            "• Claude 3.5 Sonnet (best reasoning)\n"
            "• GPT-4 Turbo (excellent all-around)\n"
            "• Llama 3.1 405B (open source powerhouse)\n"
            "• Gemini Pro 1.5 (large context)\n"
            "• And many more!",
            title="🔑 API Key Setup Required"
        ))
        return None
    
    # Initialize OpenRouter
    config = OpenRouterConfig(
        api_key=api_key,
        default_model="anthropic/claude-3.5-sonnet"  # Best reasoning model
    )
    
    return OpenRouterIntegration(config)


async def show_available_models(openrouter: OpenRouterIntegration):
    """Display available models."""
    console.print("\n[bold blue]Fetching available models...[/bold blue]")
    
    models = await openrouter.get_available_models()
    
    if not models:
        console.print("[red]No models available or API error[/red]")
        return
    
    # Create table of top models
    table = Table(title="🤖 Top Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Context", style="yellow")
    table.add_column("Best For", style="magenta")
    
    # Highlight top models
    top_models = [
        ("anthropic/claude-3.5-sonnet", "Anthropic", "200K", "Complex reasoning, analysis"),
        ("openai/gpt-4-turbo-preview", "OpenAI", "128K", "General purpose, coding"),
        ("google/gemini-pro-1.5", "Google", "1M", "Large context, research"),
        ("meta-llama/llama-3.1-405b-instruct", "Meta", "128K", "Open source, reasoning"),
        ("anthropic/claude-3-haiku", "Anthropic", "200K", "Fast, cost-effective"),
        ("openai/gpt-3.5-turbo", "OpenAI", "16K", "Quick responses, chat"),
    ]
    
    for model_id, provider, context, best_for in top_models:
        # Find model in available list
        model_info = next((m for m in models if m.id == model_id), None)
        if model_info:
            table.add_row(model_info.name, provider, context, best_for)
    
    console.print(table)
    console.print(f"\n[dim]Total available models: {len(models)}[/dim]")


async def demonstrate_model_selection(openrouter: OpenRouterIntegration):
    """Demonstrate intelligent model selection."""
    console.print("\n[bold green]🎯 Intelligent Model Selection Demo[/bold green]")
    
    # Create a complex problem
    problem = Problem(
        title="AI Ethics Framework",
        description="Design a comprehensive ethical framework for AI systems that balances innovation with safety, addresses bias and fairness concerns, ensures transparency and accountability, while being practical for implementation across different industries and scales.",
        problem_type="design",
        domain="artificial_intelligence",
        constraints=[
            "Must be implementable across industries",
            "Balance innovation with safety",
            "Address bias and fairness",
            "Ensure transparency"
        ],
        objectives=[
            "Create ethical guidelines",
            "Ensure practical implementation",
            "Address stakeholder concerns",
            "Enable responsible AI development"
        ]
    )
    
    # Initialize model selector
    selector = ModelSelector(openrouter)
    
    # Get recommendations
    console.print("Analyzing problem and selecting optimal models...")
    recommendations = await selector.recommend_model(problem)
    
    if not recommendations:
        console.print("[red]No model recommendations available[/red]")
        return
    
    # Display recommendations
    rec_table = Table(title="🎯 Model Recommendations")
    rec_table.add_column("Rank", style="cyan", width=6)
    rec_table.add_column("Model", style="green")
    rec_table.add_column("Category", style="yellow")
    rec_table.add_column("Confidence", style="magenta")
    rec_table.add_column("Reasoning", style="white")
    rec_table.add_column("Est. Cost", style="blue")
    
    for i, rec in enumerate(recommendations, 1):
        rec_table.add_row(
            f"#{i}",
            rec.model_name,
            rec.category.value.title(),
            f"{rec.confidence:.1%}",
            rec.reasoning[:50] + "..." if len(rec.reasoning) > 50 else rec.reasoning,
            f"${rec.estimated_cost:.4f}"
        )
    
    console.print(rec_table)
    
    return recommendations


async def solve_with_multiple_models(
    openrouter: OpenRouterIntegration, 
    problem: Problem, 
    model_ids: List[str]
):
    """Solve problem with multiple models and compare results."""
    console.print(f"\n[bold blue]🔄 Solving with {len(model_ids)} different models[/bold blue]")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for model_id in model_ids:
            task = progress.add_task(f"Solving with {model_id}...", total=None)
            
            solution = await openrouter.solve_problem_with_model(
                problem=problem,
                model_id=model_id,
                approach="hierarchical"
            )
            
            results.append((model_id, solution))
            progress.update(task, completed=True)
    
    # Display comparison
    comparison_table = Table(title="🏆 Model Comparison Results")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Confidence", style="green")
    comparison_table.add_column("Quality Score", style="yellow")
    comparison_table.add_column("Response Time", style="blue")
    comparison_table.add_column("Solution Preview", style="white")
    
    for model_id, solution in results:
        model_name = model_id.split("/")[-1]  # Get model name from ID
        preview = solution.solution_text[:80] + "..." if len(solution.solution_text) > 80 else solution.solution_text
        
        comparison_table.add_row(
            model_name,
            f"{solution.confidence:.1%}",
            f"{solution.get_overall_quality():.1%}",
            f"{solution.solving_time:.2f}s" if solution.solving_time else "N/A",
            preview
        )
    
    console.print(comparison_table)
    
    # Show best solution
    best_model, best_solution = max(results, key=lambda x: x[1].confidence)
    
    console.print(Panel(
        f"[bold green]Best Solution (from {best_model}):[/bold green]\n\n"
        f"{best_solution.solution_text}",
        title="🥇 Top Solution"
    ))
    
    return results


async def hybrid_reasoning_demo():
    """Demonstrate hybrid reasoning with local HRM + external models."""
    console.print(Panel(
        "[bold magenta]🔬 Hybrid Reasoning Demo[/bold magenta]\n\n"
        "This demo combines:\n"
        "• Local Hierarchical Reasoning Model (HRM)\n"
        "• External models via OpenRouter\n"
        "• Intelligent model selection\n"
        "• Multi-model comparison",
        title="Hybrid AI Reasoning"
    ))
    
    # Set up OpenRouter
    openrouter = await setup_openrouter()
    if not openrouter:
        return
    
    try:
        # Show available models
        await show_available_models(openrouter)
        
        # Demonstrate model selection
        recommendations = await demonstrate_model_selection(openrouter)
        
        if recommendations:
            # Create test problem
            problem = Problem(
                title="Sustainable Urban Planning",
                description="Design a sustainable urban development plan for a growing city of 500,000 people that reduces carbon emissions by 70% over 15 years while maintaining economic growth and improving quality of life for residents.",
                problem_type="planning",
                domain="urban_planning",
                constraints=[
                    "70% emission reduction in 15 years",
                    "Maintain economic growth",
                    "Improve quality of life",
                    "Budget constraints"
                ],
                objectives=[
                    "Environmental sustainability",
                    "Economic viability", 
                    "Social equity",
                    "Infrastructure efficiency"
                ]
            )
            
            # Use top 3 recommended models
            top_models = [rec.model_id for rec in recommendations[:3]]
            
            # Solve with multiple models
            await solve_with_multiple_models(openrouter, problem, top_models)
            
            # Compare with local HRM
            console.print("\n[bold yellow]🧠 Comparing with Local HRM[/bold yellow]")
            
            from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel
            
            local_hrm = HierarchicalReasoningModel(
                embedding_dim=256,
                use_pretrained=False  # Faster for demo
            )
            
            hrm_result = local_hrm.solve_problem(problem)
            
            console.print(f"Local HRM Confidence: {hrm_result['confidence']:.1%}")
            console.print(f"Local HRM Solutions: {len(hrm_result['solutions'])}")
            
            if hrm_result['solutions']:
                console.print(Panel(
                    f"[bold blue]Local HRM Solution:[/bold blue]\n\n"
                    f"{hrm_result['solutions'][0].solution_text}",
                    title="🧠 Local HRM Result"
                ))
        
    finally:
        await openrouter.close()


async def main():
    """Main demo function."""
    console.print(Panel(
        "[bold green]🚀 GodMode + OpenRouter Integration[/bold green]\n\n"
        "[yellow]This enhanced version combines:[/yellow]\n"
        "• Local hierarchical reasoning models\n"
        "• External API access to top LLMs\n" 
        "• Intelligent model selection\n"
        "• Multi-model comparison\n"
        "• Hybrid reasoning approaches",
        title="Enhanced GodMode System"
    ))
    
    await hybrid_reasoning_demo()
    
    console.print(Panel(
        "[bold green]✅ Demo Complete![/bold green]\n\n"
        "[yellow]Next Steps:[/yellow]\n"
        "• Set your OPENROUTER_API_KEY environment variable\n"
        "• Try different models for various problem types\n"
        "• Experiment with hybrid reasoning approaches\n"
        "• Compare local vs external model performance\n\n"
        "[cyan]Usage:[/cyan]\n"
        "python godmode_with_openrouter.py",
        title="🎯 Summary"
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Demo interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Error: {e}")