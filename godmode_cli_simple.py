#!/usr/bin/env python3
"""
GodMode - Predictive Conversation Intelligence

A simple CLI for exploring conversation prediction, future questions, and reasoning analysis.
No complex options - just ask questions and see what happens next.
"""

import asyncio
import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns
from rich.prompt import Prompt, Confirm

from godmode.core.conversation_predictor import conversation_predictor
from godmode.integrations.openrouter import OpenRouterConfig


class GodModeCLI:
    """Simple CLI for GodMode conversation prediction."""

    def __init__(self):
        self.console = Console()
        self.conversation_history: List[Dict[str, Any]] = []
        # Use demo mode by default to avoid API dependency
        from godmode.core.conversation_predictor import ConversationPredictor
        self.predictor = ConversationPredictor(demo_mode=True)

    async def run(self):
        """Main CLI loop."""
        self.show_welcome()

        # Skip OpenRouter setup in demo mode
        self.console.print("[green]✅ Demo mode enabled - no API required![/green]")

        while True:
            try:
                # Get user input
                question = Prompt.ask("\n[bold cyan]🧠 Ask anything[/bold cyan]").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    break

                if question.lower() in ['help', 'h', '?']:
                    self.show_help()
                    continue

                if question.lower() == 'history':
                    self.show_history()
                    continue

                if question.lower() == 'clear':
                    self.conversation_history.clear()
                    self.console.print("[green]✅ Conversation history cleared[/green]")
                    continue

                # Predict conversation
                with self.console.status("[bold blue]🔮 Analyzing your question...", spinner="dots"):
                    result = await self.predictor.predict_conversation(
                        current_question=question,
                        conversation_context=self.conversation_history[-5:] if self.conversation_history else None
                    )

                # Display results
                await self.display_prediction_results(result)

                # Add to history
                self.conversation_history.append({
                    'question': question,
                    'timestamp': datetime.now(),
                    'prediction': result
                })

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error: {e}[/red]")
                continue

        self.show_goodbye()

    def show_welcome(self):
        """Show welcome message."""
        welcome = Panel(
            "[bold white]Welcome to GodMode[/bold white]\n\n"
            "[cyan]Predictive Conversation Intelligence[/cyan]\n\n"
            "Ask any question and see:\n"
            "• 🔮 The next 30 questions in your conversation\n"
            "• 🌳 Alternative conversation branches\n"
            "• 🏛️ The 10 questions that led to your current thought\n"
            "• 🤖 AI-selected reasoning with explanations\n\n"
            "[dim]Type 'help' for commands, 'quit' to exit[/dim]",
            title="🧠 GodMode",
            border_style="blue"
        )
        self.console.print(welcome)

    def show_help(self):
        """Show help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[green]ask anything[/green]     - Get conversation predictions
[yellow]help[/yellow]           - Show this help
[yellow]history[/yellow]        - Show conversation history
[yellow]clear[/yellow]          - Clear conversation history
[yellow]quit[/yellow]           - Exit GodMode

[bold cyan]What GodMode Does:[/bold cyan]

🔮 [green]Future Prediction[/green]   - Predicts next 30 conversation questions
🌳 [green]Branch Exploration[/green]  - Explores alternative conversation paths
🏛️ [green]Origin Analysis[/green]     - Reveals questions leading to current thought
🤖 [green]AI Reasoning[/green]        - Automatically selects optimal reasoning approach
        """
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))

    def show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            self.console.print("[yellow]📝 No conversation history yet[/yellow]")
            return

        table = Table(title="Conversation History")
        table.add_column("Time", style="dim")
        table.add_column("Question", style="cyan")
        table.add_column("Future Questions", style="green")
        table.add_column("Branches", style="yellow")

        for item in self.conversation_history[-10:]:  # Show last 10
            prediction = item['prediction']
            table.add_row(
                item['timestamp'].strftime("%H:%M"),
                item['question'][:50] + "..." if len(item['question']) > 50 else item['question'],
                str(len(prediction.future_questions)),
                str(len(prediction.alternative_branches))
            )

        self.console.print(table)

    async def setup_openrouter(self) -> bool:
        """Setup OpenRouter integration."""
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            self.console.print("\n[yellow]⚠️  No OpenRouter API key found[/yellow]")
            api_key = Prompt.ask("Enter your OpenRouter API key", password=True)

            if not api_key:
                self.console.print("[red]❌ API key required for conversation prediction[/red]")
                return False

            # Save to environment for this session
            os.environ["OPENROUTER_API_KEY"] = api_key

        try:
            # Test the connection
            with self.console.status("[bold blue]🔗 Connecting to OpenRouter...", spinner="dots"):
                config = OpenRouterConfig(api_key=api_key)
                self.predictor = conversation_predictor.__class__(config)

                # Test with a simple question
                test_result = await self.predictor._analyze_reasoning_type("Hello world")
                if test_result['type']:
                    self.console.print("[green]✅ OpenRouter connected successfully![/green]")
                    return True
                else:
                    raise Exception("Connection test failed")

        except Exception as e:
            self.console.print(f"[red]❌ Failed to connect to OpenRouter: {e}[/red]")
            self.console.print("\n[dim]Make sure your API key is valid and you have internet connection.[/dim]")
            return False

    async def display_prediction_results(self, result):
        """Display prediction results in a beautiful format."""

        # Main question
        self.console.print(f"\n[bold cyan]🎯 Your Question:[/bold cyan] {result.current_question}")

        # AI Reasoning Analysis
        reasoning_panel = Panel(
            f"[bold yellow]Selected Reasoning:[/bold yellow] {result.selected_reasoning_type.title()}\n\n"
            f"[dim]{result.reasoning_explanation}[/dim]",
            title="🤖 AI Reasoning Analysis",
            border_style="yellow"
        )
        self.console.print(reasoning_panel)

        # Future Questions
        if result.future_questions:
            future_table = Table(title="🔮 Next 30 Questions", show_header=False)
            future_table.add_column("Question", style="green")

            for i, question in enumerate(result.future_questions[:10], 1):  # Show first 10
                future_table.add_row(f"{i}. {question}")

            if len(result.future_questions) > 10:
                future_table.add_row(f"[dim]... and {len(result.future_questions) - 10} more[/dim]")

            self.console.print(future_table)

        # Alternative Branches
        if result.alternative_branches:
            branches_table = Table(title="🌳 Alternative Branches")
            branches_table.add_column("Branch", style="yellow")
            branches_table.add_column("Path", style="cyan")
            branches_table.add_column("Probability", style="magenta")

            for branch in result.alternative_branches[:5]:  # Show first 5
                branches_table.add_row(
                    branch.key_decision_points[0] if branch.key_decision_points else "Alternative",
                    branch.reasoning_path[:80] + "..." if len(branch.reasoning_path) > 80 else branch.reasoning_path,
                    f"{branch.probability:.1%}"
                )

            self.console.print(branches_table)

        # Origin Questions
        if result.origin_questions:
            origin_table = Table(title="🏛️ Origin Questions", show_header=False)
            origin_table.add_column("Question", style="blue")

            for i, question in enumerate(result.origin_questions[:10], 1):  # Show first 10
                origin_table.add_row(f"{i}. {question}")

            if len(result.origin_questions) > 10:
                origin_table.add_row(f"[dim]... and {len(result.origin_questions) - 10} more[/dim]")

            self.console.print(origin_table)

        # Confidence Score
        confidence_color = "green" if result.confidence_score > 0.8 else "yellow" if result.confidence_score > 0.6 else "red"
        confidence_panel = Panel(
            f"[bold {confidence_color}]Confidence: {result.confidence_score:.1%}[/bold {confidence_color}]\n\n"
            f"[dim]This score represents how certain the AI is about these predictions based on the conversation context and reasoning quality.[/dim]",
            title="📊 Prediction Confidence",
            border_style=confidence_color
        )
        self.console.print(confidence_panel)

        # Metadata
        if result.prediction_metadata:
            meta_text = f"Model: {result.prediction_metadata.get('model_used', 'Unknown')} | "
            meta_text += f"Questions: {len(result.future_questions)} | "
            meta_text += f"Branches: {len(result.alternative_branches)} | "
            meta_text += f"Origins: {len(result.origin_questions)}"

            self.console.print(f"\n[dim]📋 {meta_text}[/dim]")

    def show_goodbye(self):
        """Show goodbye message."""
        goodbye = Panel(
            "[bold green]Thank you for using GodMode![/bold green]\n\n"
            f"[cyan]Session Summary:[/cyan]\n"
            f"• Questions asked: {len(self.conversation_history)}\n"
            f"• Total predictions: {sum(len(item['prediction'].future_questions) for item in self.conversation_history)}\n"
            f"• Branches explored: {sum(len(item['prediction'].alternative_branches) for item in self.conversation_history)}\n\n"
            "[dim]Your conversation intelligence awaits your return...[/dim]",
            title="👋 Goodbye",
            border_style="green"
        )
        self.console.print(goodbye)


async def main():
    """Main entry point."""
    cli = GodModeCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
