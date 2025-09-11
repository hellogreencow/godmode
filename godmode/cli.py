"""Command-line interface for GODMODE."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.json import JSON

from .core.engine import GodmodeEngine
from .models.commands import InitCommand, AdvanceCommand, ContinueCommand, SummarizeCommand, Budgets
from .models.responses import GodmodeResponse
from .tests.mocks import MockModelOrchestrator


app = typer.Typer(
    name="godmode",
    help="GODMODE - Superhuman Question Foresight Engine",
    add_completion=False
)
console = Console()


class GodmodeCLI:
    """CLI wrapper for GODMODE engine."""
    
    def __init__(self, use_mock_models: bool = True):
        self.engine = GodmodeEngine()
        self.use_mock_models = use_mock_models
        self.mock_orchestrator = MockModelOrchestrator() if use_mock_models else None
        self.session_history: list[GodmodeResponse] = []
        self.current_response: Optional[GodmodeResponse] = None
    
    async def process_init_command(
        self, 
        question: str, 
        context: Optional[str] = None,
        budgets: Optional[Budgets] = None
    ) -> GodmodeResponse:
        """Process INIT command."""
        command = InitCommand(
            current_question=question,
            context=context,
            budgets=budgets
        )
        
        if self.use_mock_models:
            # Inject mock models into engine for testing
            response = await self._process_with_mocks(command)
        else:
            response = await self.engine.process_command(command)
        
        self.session_history.append(response)
        self.current_response = response
        return response
    
    async def process_advance_command(
        self, 
        node_id: str, 
        user_answer: Optional[str] = None
    ) -> GodmodeResponse:
        """Process ADVANCE command."""
        if not self.current_response:
            raise ValueError("No active session. Run 'init' command first.")
        
        command = AdvanceCommand(
            node_id=node_id,
            user_answer=user_answer
        )
        
        response = await self.engine.process_command(command)
        self.session_history.append(response)
        self.current_response = response
        return response
    
    async def process_continue_command(self, thread_id: str) -> GodmodeResponse:
        """Process CONTINUE command."""
        if not self.current_response:
            raise ValueError("No active session. Run 'init' command first.")
        
        command = ContinueCommand(thread_id=thread_id)
        
        response = await self.engine.process_command(command)
        self.session_history.append(response)
        self.current_response = response
        return response
    
    async def process_summarize_command(self, thread_id: str) -> GodmodeResponse:
        """Process SUMMARIZE command."""
        if not self.current_response:
            raise ValueError("No active session. Run 'init' command first.")
        
        command = SummarizeCommand(thread_id=thread_id)
        
        response = await self.engine.process_command(command)
        return response
    
    async def _process_with_mocks(self, command: InitCommand) -> GodmodeResponse:
        """Process command using mock models."""
        # Use mock orchestrator to generate structured questions
        mock_questions = await self.mock_orchestrator.enumerate_and_rank(
            command.current_question,
            command.context
        )
        
        # Convert to proper response format (simplified for demo)
        from .models.responses import GraphUpdate, OntologyUpdate, Meta
        from .models.core import Lane, Thread, ThreadStatus
        
        # Create mock priors (first 2 questions)
        priors = mock_questions[:2]
        
        # Create mock scenarios (remaining questions split into lanes)
        remaining_questions = mock_questions[2:]
        scenarios = []
        
        if remaining_questions:
            # Split into 3 lanes
            lane_size = max(1, len(remaining_questions) // 3)
            
            for i in range(3):
                start_idx = i * lane_size
                end_idx = start_idx + lane_size if i < 2 else len(remaining_questions)
                lane_questions = remaining_questions[start_idx:end_idx]
                
                if lane_questions:
                    lane = Lane(
                        id=f"S-{chr(65+i)}",
                        name=f"Scenario {chr(65+i)}",
                        description=f"Mock scenario lane {i+1}",
                        lane=lane_questions
                    )
                    scenarios.append(lane)
        
        # Create mock thread
        threads = [
            Thread(
                thread_id="T1",
                origin_node_id=priors[0].id if priors else "Q1",
                path=[priors[0].id] if priors else ["Q1"],
                status=ThreadStatus.ACTIVE,
                summary="Mock thread starting from first prior"
            )
        ]
        
        graph_update = GraphUpdate(
            current_question=command.current_question,
            priors=priors,
            scenarios=scenarios,
            threads=threads,
            meta=Meta(
                budgets_used=command.budgets.dict() if command.budgets else {},
                notes="Generated using mock models"
            )
        )
        
        ontology_update = OntologyUpdate(
            entities=[],
            relations=[],
            mappings=[]
        )
        
        # Generate chat reply
        chat_reply = f"Analyzed '{command.current_question}' with {len(priors)} priors and {len(scenarios)} scenario lanes."
        
        return GodmodeResponse(
            chat_reply=chat_reply,
            graph_update=graph_update,
            ontology_update=ontology_update
        )
    
    def display_response(self, response: GodmodeResponse) -> None:
        """Display a GODMODE response in the terminal."""
        console.print(Panel(
            response.chat_reply,
            title="🧠 GODMODE Response",
            border_style="blue"
        ))
        
        # Display priors tree
        if response.graph_update.priors:
            self._display_priors_tree(response.graph_update.priors)
        
        # Display scenario lanes
        if response.graph_update.scenarios:
            self._display_scenarios_table(response.graph_update.scenarios)
        
        # Display threads
        if response.graph_update.threads:
            self._display_threads_table(response.graph_update.threads)
        
        # Display performance stats if using mocks
        if self.mock_orchestrator:
            self._display_performance_stats()
    
    def _display_priors_tree(self, priors) -> None:
        """Display priors as a tree structure."""
        console.print("\n📋 Prior Questions (Backward Reasoning):")
        
        tree = Tree("🎯 Prerequisites")
        
        # Group by level
        by_level = {}
        for prior in priors:
            level = prior.level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(prior)
        
        # Build tree structure
        level_nodes = {}
        for level in sorted(by_level.keys()):
            for prior in by_level[level]:
                node_text = f"[bold]{prior.id}[/bold]: {prior.text[:60]}..."
                node_text += f" [dim]({prior.cognitive_move.value}, gain={prior.expected_info_gain:.2f})[/dim]"
                
                if level == 1:
                    level_nodes[prior.id] = tree.add(node_text)
                else:
                    # Find parent node
                    parent_node = tree
                    for parent_id in prior.builds_on:
                        if parent_id in level_nodes:
                            parent_node = level_nodes[parent_id]
                            break
                    level_nodes[prior.id] = parent_node.add(node_text)
        
        console.print(tree)
    
    def _display_scenarios_table(self, scenarios) -> None:
        """Display scenario lanes as a table."""
        console.print("\n🚀 Future Scenarios:")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Lane", style="cyan", width=8)
        table.add_column("Name", style="green", width=15)
        table.add_column("Description", width=25)
        table.add_column("Questions", width=40)
        
        for scenario in scenarios:
            questions_text = ""
            for q in scenario.lane[:3]:  # Show first 3 questions
                questions_text += f"• {q.id}: {q.text[:30]}...\n"
            if len(scenario.lane) > 3:
                questions_text += f"... and {len(scenario.lane) - 3} more"
            
            table.add_row(
                scenario.id,
                scenario.name,
                scenario.description,
                questions_text.strip()
            )
        
        console.print(table)
    
    def _display_threads_table(self, threads) -> None:
        """Display active threads."""
        console.print("\n🧵 Active Threads:")
        
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Thread ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Path Length", style="magenta")
        table.add_column("Summary", width=50)
        
        for thread in threads:
            table.add_row(
                thread.thread_id,
                thread.status.value,
                str(len(thread.path)),
                thread.summary
            )
        
        console.print(table)
    
    def _display_performance_stats(self) -> None:
        """Display performance statistics."""
        if not self.mock_orchestrator:
            return
        
        stats = self.mock_orchestrator.get_performance_stats()
        
        console.print("\n📊 Performance Stats:")
        perf_table = Table(show_header=True, header_style="bold blue")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_table.add_row("Total Calls", str(stats["total_calls"]))
        perf_table.add_row("Avg Latency", f"{stats['avg_latency']:.3f}s")
        perf_table.add_row("Enumeration Calls", str(stats["model_calls"]["enumeration"]))
        perf_table.add_row("Reranking Calls", str(stats["model_calls"]["reranking"]))
        perf_table.add_row("Stitching Calls", str(stats["model_calls"]["stitching"]))
        
        console.print(perf_table)
    
    def save_session(self, filepath: Path) -> None:
        """Save current session to file."""
        session_data = {
            "history": [response.dict() for response in self.session_history],
            "current_response": self.current_response.dict() if self.current_response else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        console.print(f"✅ Session saved to {filepath}")
    
    def load_session(self, filepath: Path) -> None:
        """Load session from file."""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        # Note: In a real implementation, we'd need to properly deserialize
        # the Pydantic models from the JSON data
        console.print(f"✅ Session loaded from {filepath}")
        console.print(f"Found {len(session_data.get('history', []))} responses in history")


# CLI instance
cli = GodmodeCLI()


@app.command()
def init(
    question: str = typer.Argument(..., help="The question to analyze"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Optional context"),
    beam_width: int = typer.Option(4, "--beam-width", "-b", help="Beam width for candidate generation"),
    depth_back: int = typer.Option(4, "--depth-back", help="Depth for backward reasoning"),
    depth_fwd: int = typer.Option(5, "--depth-forward", help="Depth for forward reasoning"),
    save_to: Optional[str] = typer.Option(None, "--save", "-s", help="Save session to file")
):
    """Initialize GODMODE with a question."""
    budgets = Budgets(
        beam_width=beam_width,
        depth_back=depth_back,
        depth_fwd=depth_fwd
    )
    
    async def run_init():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🧠 Analyzing question...", total=None)
            
            try:
                response = await cli.process_init_command(question, context, budgets)
                progress.update(task, description="✅ Analysis complete!")
                
                cli.display_response(response)
                
                if save_to:
                    cli.save_session(Path(save_to))
                
            except Exception as e:
                progress.update(task, description="❌ Analysis failed!")
                console.print(f"[red]Error: {str(e)}[/red]")
                sys.exit(1)
    
    asyncio.run(run_init())


@app.command()
def advance(
    node_id: str = typer.Argument(..., help="Node ID to advance from"),
    answer: Optional[str] = typer.Option(None, "--answer", "-a", help="User answer to incorporate")
):
    """Advance from a specific node."""
    async def run_advance():
        try:
            response = await cli.process_advance_command(node_id, answer)
            cli.display_response(response)
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_advance())


@app.command()
def continue_thread(
    thread_id: str = typer.Argument(..., help="Thread ID to continue")
):
    """Continue a specific thread."""
    async def run_continue():
        try:
            response = await cli.process_continue_command(thread_id)
            cli.display_response(response)
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_continue())


@app.command()
def summarize(
    thread_id: str = typer.Argument(..., help="Thread ID to summarize")
):
    """Get summary of a thread."""
    async def run_summarize():
        try:
            response = await cli.process_summarize_command(thread_id)
            console.print(Panel(
                response.chat_reply,
                title=f"📝 Thread {thread_id} Summary",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_summarize())


@app.command()
def interactive():
    """Start interactive GODMODE session."""
    console.print(Panel(
        "🧠 GODMODE Interactive Session\n\n"
        "Commands:\n"
        "  init <question>     - Start with a question\n"
        "  advance <node_id>   - Advance from a node\n"
        "  continue <thread>   - Continue a thread\n"
        "  summarize <thread>  - Summarize a thread\n"
        "  show                - Show current response\n"
        "  save <file>         - Save session\n"
        "  load <file>         - Load session\n"
        "  quit                - Exit session",
        title="Welcome to GODMODE",
        border_style="blue"
    ))
    
    while True:
        try:
            command = Prompt.ask("\n🧠 GODMODE", default="help")
            
            if command.lower() in ["quit", "exit", "q"]:
                console.print("👋 Goodbye!")
                break
            elif command.lower() in ["help", "h"]:
                console.print("Available commands: init, advance, continue, summarize, show, save, load, quit")
            elif command.lower() == "show":
                if cli.current_response:
                    cli.display_response(cli.current_response)
                else:
                    console.print("[yellow]No active response. Use 'init' to start.[/yellow]")
            elif command.startswith("init "):
                question = command[5:].strip()
                if question:
                    async def run_init():
                        response = await cli.process_init_command(question)
                        cli.display_response(response)
                    asyncio.run(run_init())
                else:
                    console.print("[red]Please provide a question after 'init'[/red]")
            elif command.startswith("advance "):
                node_id = command[8:].strip()
                if node_id:
                    async def run_advance():
                        response = await cli.process_advance_command(node_id)
                        cli.display_response(response)
                    asyncio.run(run_advance())
                else:
                    console.print("[red]Please provide a node ID after 'advance'[/red]")
            elif command.startswith("save "):
                filepath = command[5:].strip()
                if filepath and cli.current_response:
                    cli.save_session(Path(filepath))
                else:
                    console.print("[red]Please provide a filename and ensure you have an active session[/red]")
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


@app.command()
def demo():
    """Run a demo session with sample questions."""
    sample_questions = [
        "How should we improve customer satisfaction for our SaaS product?",
        "What's the best way to scale our engineering team from 10 to 50 people?",
        "How can we increase revenue by 50% in the next 12 months?",
        "What strategy should we use to enter the European market?"
    ]
    
    console.print(Panel(
        "🎯 GODMODE Demo\n\n"
        "This demo will show GODMODE analyzing sample business questions.",
        title="Demo Mode",
        border_style="green"
    ))
    
    for i, question in enumerate(sample_questions, 1):
        console.print(f"\n[bold blue]Demo Question {i}:[/bold blue] {question}")
        
        if not Confirm.ask("Analyze this question?", default=True):
            continue
        
        async def run_demo():
            response = await cli.process_init_command(question)
            cli.display_response(response)
        
        asyncio.run(run_demo())
        
        if i < len(sample_questions):
            if not Confirm.ask("Continue to next question?", default=True):
                break
    
    console.print("\n🎉 Demo complete! Use 'godmode interactive' for a full session.")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()