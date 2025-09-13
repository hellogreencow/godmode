"""
Command-line interface for the GodMode hierarchical reasoning system.

This module provides a comprehensive CLI for interacting with all aspects
of the GodMode system, including reasoning, memory management, and web interface.
"""

import asyncio
import json
import logging
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from godmode.core.engine import GodModeEngine
from godmode.experimental.hierarchical_reasoning import HierarchicalReasoningModel
from godmode.models.core import Problem, ReasoningType, CognitiveLevel
from godmode.web.app import GodModeWebApp


app = typer.Typer(
    name="godmode",
    help="🧠 GodMode: Advanced Hierarchical Reasoning System",
    add_completion=False,
)

console = Console()

# Global instances
engine: Optional[GodModeEngine] = None
hierarchical_model: Optional[HierarchicalReasoningModel] = None


def find_available_port(avoid_ranges: List[tuple] = None, preferred_ranges: List[tuple] = None) -> int:
    """Find an available port with intelligent allocation."""
    import random

    # Default ranges to avoid (8000-8999, 3000-3999 for common dev servers)
    if avoid_ranges is None:
        avoid_ranges = [(8000, 8999), (3000, 3999)]

    # Preferred ranges (high ports, avoiding common conflicts)
    if preferred_ranges is None:
        preferred_ranges = [(10000, 19999), (20000, 29999), (40000, 49999)]

    # Flatten avoid ranges into a set for quick lookup
    avoid_ports = set()
    for start, end in avoid_ranges:
        avoid_ports.update(range(start, end + 1))

    # Try preferred ranges first
    for start, end in preferred_ranges:
        ports_to_try = list(range(start, end + 1))
        random.shuffle(ports_to_try)

        for port in ports_to_try:
            if port in avoid_ports:
                continue

            if is_port_available(port):
                return port

    # Fallback: try random high ports
    for _ in range(100):
        port = random.randint(10000, 65535)
        if port not in avoid_ports and is_port_available(port):
            return port

    raise RuntimeError("Could not find an available port after 100 attempts")


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            return result != 0  # 0 means connection successful (port in use)
    except:
        return False


class PortManager:
    """Manages port allocation for GodMode services."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / ".godmode_ports.json"
        self.ports = self._load_ports()

    def _load_ports(self) -> Dict[str, int]:
        """Load current port assignments."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_ports(self):
        """Save current port assignments."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.ports, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]⚠️ Warning: Could not save port configuration: {e}[/yellow]")

    def get_port(self, service_name: str, force_new: bool = False) -> int:
        """Get a port for a service, allocating a new one if needed."""
        if not force_new and service_name in self.ports:
            port = self.ports[service_name]
            if is_port_available(port):
                return port

        # Find a new port
        port = find_available_port()
        self.ports[service_name] = port
        self._save_ports()
        return port

    def release_port(self, service_name: str):
        """Release a port assignment."""
        if service_name in self.ports:
            del self.ports[service_name]
            self._save_ports()

    def list_ports(self) -> Dict[str, int]:
        """List all current port assignments."""
        return self.ports.copy()


# Global port manager
port_manager = PortManager()


def init_engine(
    memory_size: int = 10000,
    reasoning_depth: int = 5,
    enable_gpu: bool = True,
    verbose: bool = False,
) -> GodModeEngine:
    """Initialize the GodMode engine."""
    global engine
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    if engine is None:
        with console.status("[bold blue]Initializing GodMode engine..."):
            engine = GodModeEngine(
                memory_size=memory_size,
                reasoning_depth=reasoning_depth,
                enable_gpu=enable_gpu,
            )
        console.print("✅ GodMode engine initialized", style="green")
    
    return engine


def init_hierarchical_model(
    embedding_dim: int = 512,
    use_pretrained: bool = True,
    verbose: bool = False,
) -> HierarchicalReasoningModel:
    """Initialize the hierarchical reasoning model."""
    global hierarchical_model
    
    if hierarchical_model is None:
        with console.status("[bold blue]Loading hierarchical reasoning model..."):
            hierarchical_model = HierarchicalReasoningModel(
                embedding_dim=embedding_dim,
                use_pretrained=use_pretrained,
            )
        console.print("✅ Hierarchical model loaded", style="green")
    
    return hierarchical_model


@app.command()
def solve(
    problem: str = typer.Argument(..., help="Problem description to solve"),
    reasoning_type: ReasoningType = typer.Option(
        ReasoningType.HIERARCHICAL,
        "--type", "-t",
        help="Type of reasoning to use"
    ),
    max_time: Optional[float] = typer.Option(None, "--time", help="Maximum time in seconds"),
    min_confidence: float = typer.Option(0.7, "--confidence", help="Minimum confidence threshold"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    use_hierarchical: bool = typer.Option(False, "--hierarchical", help="Use experimental hierarchical model"),
):
    """Solve a problem using the reasoning engine."""
    
    console.print(Panel(f"[bold blue]Problem:[/bold blue] {problem}", title="🧠 GodMode Reasoning"))
    
    if use_hierarchical:
        model = init_hierarchical_model(verbose=verbose)
        
        with console.status("[bold yellow]Solving with hierarchical model..."):
            result = model.solve_problem(problem)
        
        # Display results
        console.print("\n[bold green]Solutions Found:[/bold green]")
        for i, solution in enumerate(result["solutions"], 1):
            console.print(f"\n[bold cyan]Solution {i}:[/bold cyan]")
            console.print(Panel(solution.solution_text))
            console.print(f"Confidence: {solution.confidence:.1%}")
        
        console.print(f"\n[bold blue]Overall Confidence:[/bold blue] {result['confidence']:.1%}")
        
        if verbose:
            console.print("\n[bold yellow]Reasoning Trace:[/bold yellow]")
            rprint(result["reasoning_trace"])
    
    else:
        engine = init_engine(verbose=verbose)
        
        async def solve_async():
            return await engine.solve_problem(
                problem=problem,
                reasoning_type=reasoning_type,
                max_time=max_time,
                min_confidence=min_confidence,
            )
        
        with console.status("[bold yellow]Reasoning in progress..."):
            response = asyncio.run(solve_async())
        
        # Display results
        if response.solution:
            console.print("\n[bold green]Solution Found:[/bold green]")
            console.print(Panel(response.solution.solution_text))
            console.print(f"Confidence: {response.solution.confidence:.1%}")
            console.print(f"Quality: {response.solution.get_overall_quality():.1%}")
        else:
            console.print("[bold red]No solution found[/bold red]")
        
        if response.alternative_solutions:
            console.print(f"\n[bold cyan]Alternative Solutions ({len(response.alternative_solutions)}):[/bold cyan]")
            for i, alt_solution in enumerate(response.alternative_solutions, 1):
                console.print(f"{i}. {alt_solution.solution_text[:100]}...")
        
        if verbose and response.reasoning_trace:
            console.print(f"\n[bold yellow]Reasoning Steps:[/bold yellow] {response.total_steps}")
            console.print(f"[bold yellow]Success Rate:[/bold yellow] {response.success_rate:.1%}")
            console.print(f"[bold yellow]Quality Scores:[/bold yellow]")
            console.print(f"  Coherence: {response.reasoning_coherence:.1%}")
            console.print(f"  Consistency: {response.reasoning_consistency:.1%}")
            console.print(f"  Efficiency: {response.reasoning_efficiency:.1%}")
    
    # Save results if requested
    if output_file:
        results = {
            "problem": problem,
            "solution": response.solution.model_dump() if hasattr(response, 'solution') and response.solution else None,
            "confidence": result["confidence"] if use_hierarchical else (response.solution.confidence if response.solution else 0),
            "reasoning_type": reasoning_type.value if not use_hierarchical else "hierarchical_experimental",
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"✅ Results saved to {output_file}")


@app.command()
def experiment(
    model: str = typer.Option("hrm", help="Experimental model to use (hrm, attention, architecture)"),
    problem: str = typer.Option("", help="Problem to solve (interactive if not provided)"),
    iterations: int = typer.Option(1, help="Number of iterations to run"),
    compare: bool = typer.Option(False, help="Compare with standard reasoning"),
):
    """Run experimental hierarchical reasoning models."""
    
    console.print(Panel("[bold magenta]🧪 Experimental Hierarchical Reasoning[/bold magenta]", 
                       title="Experiment Mode"))
    
    if not problem:
        problem = Prompt.ask("Enter a complex problem to solve")
    
    hierarchical_model = init_hierarchical_model()
    
    results = []
    
    for i in range(iterations):
        console.print(f"\n[bold blue]Iteration {i+1}/{iterations}[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running hierarchical reasoning...", total=None)
            
            result = hierarchical_model.solve_problem(problem)
            results.append(result)
            
            progress.update(task, completed=True)
        
        console.print(f"Confidence: {result['confidence']:.1%}")
        
        if len(result['solutions']) > 0:
            console.print("✅ Solution generated")
        else:
            console.print("❌ No solution found")
    
    # Summary statistics
    if iterations > 1:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['solutions']) / len(results)
        
        console.print(f"\n[bold green]Experiment Summary:[/bold green]")
        console.print(f"Average Confidence: {avg_confidence:.1%}")
        console.print(f"Success Rate: {success_rate:.1%}")
    
    # Compare with standard reasoning if requested
    if compare:
        console.print("\n[bold yellow]Comparing with standard reasoning...[/bold yellow]")
        engine = init_engine()
        
        async def compare_async():
            return await engine.solve_problem(problem=problem)
        
        standard_result = asyncio.run(compare_async())
        
        table = Table(title="Comparison Results")
        table.add_column("Method", style="cyan")
        table.add_column("Confidence", style="green")
        table.add_column("Solution Found", style="yellow")
        
        table.add_row(
            "Hierarchical Model",
            f"{results[0]['confidence']:.1%}",
            "✅" if results[0]['solutions'] else "❌"
        )
        table.add_row(
            "Standard Engine", 
            f"{standard_result.solution.confidence:.1%}" if standard_result.solution else "0%",
            "✅" if standard_result.solution else "❌"
        )
        
        console.print(table)


@app.command()
def web(
    port: Optional[int] = typer.Option(None, help="Port to run the web server on (auto-assigned if not specified)"),
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
):
    """Launch the web interface."""

    # Auto-assign port if not specified
    if port is None:
        try:
            port = port_manager.get_port("web_server")
            console.print(f"[green]🔀 Auto-assigned port: {port} (managed by PortManager)[/green]")
        except RuntimeError as e:
            console.print(f"[red]❌ {e}[/red]")
            return

    console.print(Panel(
        f"[bold green]🌐 Starting GodMode Web Interface[/bold green]\n\n"
        f"🌐 URL: http://{host}:{port}\n"
        f"🧪 Experiment: http://localhost:{port}/experiment\n"
        f"📊 Demo: http://localhost:{port}/demo\n"
        f"📈 Analytics: http://localhost:{port}/analytics\n"
        f"📖 API docs: http://localhost:{port}/api/docs\n\n"
        f"[dim]💡 Port {port} was auto-assigned to avoid conflicts[/dim]",
        title="Web Server"
    ))
    
    engine = init_engine(verbose=debug)
    hierarchical_model = init_hierarchical_model(verbose=debug)
    
    web_app = GodModeWebApp(
        engine=engine,
        hierarchical_model=hierarchical_model,
        debug=debug,
    )
    
    try:
        web_app.run(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        console.print("\n👋 Shutting down web server...")


@app.command()
def ports(
    action: str = typer.Argument(..., help="Action: list, release, reset"),
    service: Optional[str] = typer.Option(None, help="Service name for release action"),
):
    """Manage port assignments for GodMode services."""
    if action == "list":
        current_ports = port_manager.list_ports()
        if current_ports:
            table = Table(title="GodMode Port Assignments")
            table.add_column("Service", style="cyan")
            table.add_column("Port", style="green")
            table.add_column("Status", style="yellow")

            for service_name, port in current_ports.items():
                status = "✅ Available" if is_port_available(port) else "❌ In use"
                table.add_row(service_name, str(port), status)

            console.print(table)
        else:
            console.print("[dim]No port assignments found.[/dim]")

    elif action == "release":
        if not service:
            console.print("[red]❌ Service name required for release action[/red]")
            return

        port_manager.release_port(service)
        console.print(f"[green]✅ Released port for service: {service}[/green]")

    elif action == "reset":
        # Reset all port assignments
        import os
        config_file = Path.home() / ".godmode_ports.json"
        if config_file.exists():
            try:
                os.remove(config_file)
                console.print("[green]✅ Reset all port assignments[/green]")
            except Exception as e:
                console.print(f"[red]❌ Failed to reset ports: {e}[/red]")
        else:
            console.print("[dim]No port configuration file found.[/dim]")

    else:
        console.print("[red]❌ Invalid action. Use: list, release, reset[/red]")


@app.command()
def port_test():
    """Test the intelligent port allocation system."""
    console.print("[bold green]🧪 GODMODE INTELLIGENT PORT ALLOCATION TEST[/bold green]")
    console.print("=" * 60)

    # Test port allocation
    console.print("\n[cyan]1. Testing port allocation...[/cyan]")
    test_ports = {}

    services = ["web_main", "api_gateway", "websocket_hub", "cache_server", "metrics_collector"]
    for service in services:
        port = port_manager.get_port(service)
        test_ports[service] = port
        console.print(f"   ✅ {service}: {port}")

    # Show current assignments
    console.print("\n[cyan]2. Current port assignments:[/cyan]")
    current_ports = port_manager.list_ports()
    for service, port in current_ports.items():
        status = "✅ Available" if is_port_available(port) else "❌ In use"
        console.print(f"   {service}: {port} ({status})")

    # Test port conflicts
    console.print("\n[cyan]3. Testing conflict avoidance:[/cyan]")
    avoid_ranges = [(8000, 8999), (3000, 3999)]
    avoid_ports = set()
    for start, end in avoid_ranges:
        avoid_ports.update(range(start, end + 1))

    all_good = True
    for service, port in test_ports.items():
        if port in avoid_ports:
            console.print(f"   ❌ {service} got forbidden port: {port}")
            all_good = False
        else:
            console.print(f"   ✅ {service} avoided forbidden ranges: {port}")

    # Test persistence
    console.print("\n[cyan]4. Testing persistence:[/cyan]")
    console.print("   Creating new PortManager instance...")
    new_manager = PortManager()
    persisted_ports = new_manager.list_ports()

    if persisted_ports == current_ports:
        console.print("   ✅ Port assignments persisted correctly")
    else:
        console.print("   ❌ Port assignments not persisted")
        all_good = False

    # Summary
    console.print("\n" + "=" * 60)
    if all_good:
        console.print("[bold green]🎉 ALL TESTS PASSED![/bold green]")
        console.print("[green]✅ Intelligent port allocation working perfectly")
        console.print("[green]✅ Conflict avoidance functioning")
        console.print("[green]✅ Persistence working")
        console.print("[green]✅ No ports in forbidden ranges (8000-8999, 3000-3999)[/green]")
    else:
        console.print("[bold red]❌ SOME TESTS FAILED[/bold red]")

    console.print("\n[dim]💡 Port configuration saved to: ~/.godmode_ports.json[/dim]")
    console.print("[dim]💡 Use 'godmode ports list' to view current assignments[/dim]")
    console.print("[dim]💡 Use 'godmode ports release <service>' to free ports[/dim]")


@app.command()
def memory(
    action: str = typer.Argument(..., help="Action: store, retrieve, stats, consolidate"),
    query: Optional[str] = typer.Option(None, help="Query for retrieve action"),
    content: Optional[str] = typer.Option(None, help="Content for store action"),
    memory_type: str = typer.Option("working", help="Memory type: working, long_term, episodic"),
    top_k: int = typer.Option(5, help="Number of results for retrieve"),
):
    """Interact with the memory system."""
    
    engine = init_engine()
    
    async def memory_async():
        if action == "store":
            if not content:
                content = Prompt.ask("Enter content to store")
            
            item_id = await engine.memory.store(
                content={"text": content, "type": "user_input"},
                memory_type=memory_type,
            )
            console.print(f"✅ Stored with ID: {item_id}")
        
        elif action == "retrieve":
            if not query:
                query = Prompt.ask("Enter search query")
            
            results = await engine.memory.retrieve(
                query=query,
                memory_types=[memory_type] if memory_type != "all" else None,
                top_k=top_k,
            )
            
            if results:
                table = Table(title=f"Memory Search Results ({len(results)})")
                table.add_column("ID", style="cyan")
                table.add_column("Type", style="yellow")
                table.add_column("Content", style="white")
                table.add_column("Activation", style="green")
                
                for item in results:
                    content_preview = str(item.content)[:50] + "..." if len(str(item.content)) > 50 else str(item.content)
                    table.add_row(
                        str(item.id)[:8],
                        item.memory_type,
                        content_preview,
                        f"{item.get_activation():.2f}"
                    )
                
                console.print(table)
            else:
                console.print("No results found")
        
        elif action == "stats":
            stats = engine.memory.get_statistics()
            
            table = Table(title="Memory Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in stats.items():
                table.add_row(key.replace("_", " ").title(), str(value))
            
            console.print(table)
        
        elif action == "consolidate":
            console.print("Performing memory consolidation...")
            await engine.memory._periodic_consolidation()
            console.print("✅ Consolidation complete")
        
        else:
            console.print(f"❌ Unknown action: {action}")
            console.print("Available actions: store, retrieve, stats, consolidate")
    
    asyncio.run(memory_async())


@app.command()
def stats(
    format: str = typer.Option("table", help="Output format: table, json"),
    save: Optional[str] = typer.Option(None, help="Save to file"),
):
    """Show system statistics and performance metrics."""
    
    engine = init_engine()
    
    stats = engine.get_statistics()
    memory_stats = engine.memory.get_statistics()
    
    all_stats = {
        "engine": stats,
        "memory": memory_stats,
    }
    
    if format == "json":
        output = json.dumps(all_stats, indent=2, default=str)
        console.print(output)
    else:
        # Engine stats table
        engine_table = Table(title="Engine Statistics")
        engine_table.add_column("Metric", style="cyan")
        engine_table.add_column("Value", style="green")
        
        for key, value in stats.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            engine_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(engine_table)
        
        # Memory stats table
        memory_table = Table(title="Memory Statistics")
        memory_table.add_column("Metric", style="cyan")
        memory_table.add_column("Value", style="green")
        
        for key, value in memory_stats.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            memory_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(memory_table)
    
    if save:
        with open(save, 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        console.print(f"✅ Statistics saved to {save}")


@app.command()
def benchmark(
    problems: Optional[str] = typer.Option(None, help="File with problems to benchmark"),
    iterations: int = typer.Option(10, help="Number of iterations per problem"),
    compare_models: bool = typer.Option(True, help="Compare different models"),
    output: Optional[str] = typer.Option(None, help="Output file for results"),
):
    """Run benchmarks on the reasoning system."""
    
    console.print(Panel("[bold yellow]🏃 Running Benchmarks[/bold yellow]", title="Benchmark Mode"))
    
    # Default problems if none provided
    default_problems = [
        "Optimize a supply chain to minimize costs while maintaining quality",
        "Design a machine learning system that learns continuously without forgetting",
        "Plan a sustainable city that reduces emissions by 70% in 10 years",
        "Create a distributed system that is both secure and highly available",
    ]
    
    if problems:
        with open(problems, 'r') as f:
            problem_list = [line.strip() for line in f if line.strip()]
    else:
        problem_list = default_problems
    
    engine = init_engine()
    hierarchical_model = init_hierarchical_model() if compare_models else None
    
    results = []
    
    with Progress(console=console) as progress:
        task = progress.add_task("Running benchmarks...", total=len(problem_list) * iterations)
        
        for problem in problem_list:
            problem_results = {"problem": problem, "iterations": []}
            
            for i in range(iterations):
                # Standard engine
                async def benchmark_standard():
                    return await engine.solve_problem(problem=problem)
                
                start_time = asyncio.get_event_loop().time()
                standard_result = asyncio.run(benchmark_standard())
                standard_time = asyncio.get_event_loop().time() - start_time
                
                iteration_result = {
                    "iteration": i + 1,
                    "standard": {
                        "time": standard_time,
                        "confidence": standard_result.solution.confidence if standard_result.solution else 0,
                        "success": standard_result.solution is not None,
                    }
                }
                
                # Hierarchical model comparison
                if hierarchical_model:
                    start_time = asyncio.get_event_loop().time()
                    hierarchical_result = hierarchical_model.solve_problem(problem)
                    hierarchical_time = asyncio.get_event_loop().time() - start_time
                    
                    iteration_result["hierarchical"] = {
                        "time": hierarchical_time,
                        "confidence": hierarchical_result["confidence"],
                        "success": len(hierarchical_result["solutions"]) > 0,
                    }
                
                problem_results["iterations"].append(iteration_result)
                progress.advance(task)
            
            results.append(problem_results)
    
    # Display summary
    console.print("\n[bold green]Benchmark Results Summary:[/bold green]")
    
    for problem_result in results:
        console.print(f"\n[bold blue]Problem:[/bold blue] {problem_result['problem'][:60]}...")
        
        # Calculate averages
        standard_times = [it["standard"]["time"] for it in problem_result["iterations"]]
        standard_confidences = [it["standard"]["confidence"] for it in problem_result["iterations"]]
        standard_success_rate = sum(it["standard"]["success"] for it in problem_result["iterations"]) / iterations
        
        console.print(f"Standard Engine:")
        console.print(f"  Avg Time: {sum(standard_times)/len(standard_times):.3f}s")
        console.print(f"  Avg Confidence: {sum(standard_confidences)/len(standard_confidences):.1%}")
        console.print(f"  Success Rate: {standard_success_rate:.1%}")
        
        if hierarchical_model:
            hierarchical_times = [it["hierarchical"]["time"] for it in problem_result["iterations"]]
            hierarchical_confidences = [it["hierarchical"]["confidence"] for it in problem_result["iterations"]]
            hierarchical_success_rate = sum(it["hierarchical"]["success"] for it in problem_result["iterations"]) / iterations
            
            console.print(f"Hierarchical Model:")
            console.print(f"  Avg Time: {sum(hierarchical_times)/len(hierarchical_times):.3f}s")
            console.print(f"  Avg Confidence: {sum(hierarchical_confidences)/len(hierarchical_confidences):.1%}")
            console.print(f"  Success Rate: {hierarchical_success_rate:.1%}")
    
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"✅ Benchmark results saved to {output}")


@app.command()
def interactive():
    """Start an interactive reasoning session."""
    
    console.print(Panel("[bold magenta]🎮 Interactive GodMode Session[/bold magenta]", 
                       title="Interactive Mode"))
    
    engine = init_engine()
    hierarchical_model = init_hierarchical_model()
    
    console.print("Type 'help' for commands, 'quit' to exit")
    
    while True:
        try:
            command = Prompt.ask("\n[bold cyan]godmode>[/bold cyan]").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                break
            elif command == 'help':
                console.print("""
Available commands:
  solve <problem>     - Solve a problem with standard engine
  hrm <problem>       - Solve with hierarchical model
  memory <query>      - Search memory
  stats               - Show statistics
  clear               - Clear screen
  help                - Show this help
  quit                - Exit
                """)
            elif command.startswith('solve '):
                problem = command[6:]
                if problem:
                    async def solve_interactive():
                        return await engine.solve_problem(problem=problem)
                    
                    result = asyncio.run(solve_interactive())
                    if result.solution:
                        console.print(f"[green]Solution:[/green] {result.solution.solution_text}")
                        console.print(f"[blue]Confidence:[/blue] {result.solution.confidence:.1%}")
                    else:
                        console.print("[red]No solution found[/red]")
                else:
                    console.print("[red]Please provide a problem to solve[/red]")
            
            elif command.startswith('hrm '):
                problem = command[4:]
                if problem:
                    result = hierarchical_model.solve_problem(problem)
                    if result['solutions']:
                        console.print(f"[green]Solution:[/green] {result['solutions'][0].solution_text}")
                        console.print(f"[blue]Confidence:[/blue] {result['confidence']:.1%}")
                    else:
                        console.print("[red]No solution found[/red]")
                else:
                    console.print("[red]Please provide a problem to solve[/red]")
            
            elif command.startswith('memory '):
                query = command[7:]
                if query:
                    async def search_memory():
                        return await engine.memory.retrieve(query=query, top_k=3)
                    
                    results = asyncio.run(search_memory())
                    if results:
                        for i, item in enumerate(results, 1):
                            console.print(f"{i}. {item.content}")
                    else:
                        console.print("[yellow]No memory results found[/yellow]")
                else:
                    console.print("[red]Please provide a search query[/red]")
            
            elif command == 'stats':
                stats = engine.get_statistics()
                for key, value in stats.items():
                    console.print(f"{key}: {value}")
            
            elif command == 'clear':
                console.clear()
            
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n👋 Goodbye!")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
