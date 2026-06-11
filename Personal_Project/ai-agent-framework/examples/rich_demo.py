"""
Visual Demo Runner using the 'rich' library.
Generates a highly-stylized, professional terminal output representing 
the AI Agent execution loop, suitable for portfolio screenshots.
"""

import time
import sys
from pathlib import Path
from datetime import datetime

# Set console encoding to UTF-8 to support emoji on Windows standard terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.box import ROUNDED
except ImportError:
    print("Error: The 'rich' package is required for this demo.")
    print("Please install it with: pip install rich")
    sys.exit(1)


def main():
    console = Console()

    # Clear terminal for a clean screenshot
    console.clear()

    # 1. Header Banner
    banner_text = Text("\n🤖 MULTI-STEP AI AGENT FRAMEWORK 🤖\n", style="bold white")
    banner_text.append("Production-Ready Lightweight Agent Loop with Memory & Resilience\n", style="italic cyan")
    banner_text.append("Written in Python • Validated with Unittests", style="dim green")
    
    console.print(
        Panel(
            banner_text,
            title="[bold purple]Portfolio Showcase[/bold purple]",
            subtitle="[dim]v1.0.0[/dim]",
            border_style="purple",
            box=ROUNDED,
            expand=False,
            padding=(1, 4)
        )
    )
    console.print("\n")

    # 2. Agent Initialization Info
    init_table = Table(title="[bold yellow]Agent Configuration[/bold yellow]", box=ROUNDED, border_style="cyan")
    init_table.add_column("Parameter", style="bold cyan")
    init_table.add_column("Value", style="green")
    init_table.add_column("Description", style="dim white")
    
    init_table.add_row("LLM Model", "gpt-4-turbo", "Primary planning and reasoning engine")
    init_table.add_row("Memory Engine", "PersistentMemory (SQLite/JSON)", "Context preservation across sessions")
    init_table.add_row("Max Iterations", "10", "Guardrail to prevent infinite loop costs")
    init_table.add_row("Retry Strategy", "Exponential Backoff (max_retries=3)", "Network & API rate-limit resilience")
    init_table.add_row("Tool Registry", "4 registered tools", "web_search, summarize, extract_facts, save_to_db")
    
    console.print(init_table)
    console.print("\n")

    # 3. Task Input
    task_panel = Panel(
        Text("Research the latest quantum computing applications in 2026, extract key breakthroughs, and persist them in the database.", style="bold white"),
        title="[bold green]Active Task[/bold green]",
        border_style="green",
        box=ROUNDED,
        padding=(1, 2)
    )
    console.print(task_panel)
    console.print("\n")

    # 4. Agent Loop Execution (Tree visualization)
    console.print("[bold magenta]⚡ Execution Agentic Loop Trace (6-Phases)[/bold magenta]")
    
    loop_tree = Tree("[bold royal_blue1]Agent Run Session: research_session_001[/bold royal_blue1]")
    
    # Phase 1
    p1 = loop_tree.add("[bold cyan]Phase 1: OBSERVE (Context & Memory Retrieval)[/bold cyan]")
    p1.add("Loaded [green]2[/green] facts from previous sessions (Known QC experts, basic algorithms)")
    p1.add("Retrieved conversation history ([dim]Session ID: research_session_001[/dim])")
    
    # Phase 2
    p2 = loop_tree.add("[bold cyan]Phase 2: PLAN (Reasoning & Action Planning)[/bold cyan]")
    reasoning = (
        "Thinking: The user wants latest applications in 2026. I should query the web database, "
        "extract the major breakthroughs, and log them into persistent storage. Let's create a 3-step plan."
    )
    p2.add(Panel(reasoning, style="italic yellow", box=ROUNDED, title="LLM Internal Reasoning"))
    plan_table = Table(box=ROUNDED, border_style="dim yellow", show_header=True)
    plan_table.add_column("Step", style="bold yellow")
    plan_table.add_column("Tool", style="bold green")
    plan_table.add_column("Parameters", style="dim white")
    plan_table.add_column("Critical", style="bold red")
    plan_table.add_row("1", "web_search", '{"query": "quantum computing applications 2026"}', "True")
    plan_table.add_row("2", "extract_facts", '{"text": "results of step 1"}', "True")
    plan_table.add_row("3", "save_to_db", '{"record_id": "qc_2026_breakthroughs", "data": "facts"}', "True")
    p2.add(plan_table)
    
    # Phase 3
    p3 = loop_tree.add("[bold cyan]Phase 3: EXECUTE (Tool Invocation & Resiliency)[/bold cyan]")
    p3.add("Executing tool [bold green]web_search[/bold green]... [bold green]✓ SUCCESS[/bold green] (0.45s)")
    p3.add("Executing tool [bold green]extract_facts[/bold green]... [bold green]✓ SUCCESS[/bold green] (0.21s)")
    
    # Simulate a temporary failure and retry
    db_node = p3.add("Executing tool [bold green]save_to_db[/bold green]...")
    db_node.add("[bold red]✗ FAILED[/bold red] - Database connection timeout. Retrying in [yellow]1.0s[/yellow]...")
    db_node.add("[bold green]✓ SUCCESS[/bold green] - Attempt 2 successful (0.12s)")
    
    # Phase 4
    p4 = loop_tree.add("[bold cyan]Phase 4: REFLECT (Evaluation & Fact Learning)[/bold cyan]")
    p4.add("Saved [green]3[/green] new facts to Persistent Memory Store:")
    p4.add("  ➔ [bold green]qc_fact_1[/bold green]: Quantum supremacy achieved in chemistry simulation")
    p4.add("  ➔ [bold green]qc_fact_2[/bold green]: Room-temperature superconductor qubits validation")
    
    # Phase 5
    p5 = loop_tree.add("[bold cyan]Phase 5: DECIDE (Loop Termination Assessment)[/bold cyan]")
    p5.add("Evaluation: [green]All steps completed successfully. Goals achieved.[/green]")
    p5.add("Action: [bold red]TERMINATE_LOOP[/bold red]")
    
    # Phase 6
    p6 = loop_tree.add("[bold cyan]Phase 6: RETURN (Output Packaging)[/bold cyan]")
    p6.add("Assembling final structured payload for caller.")

    console.print(loop_tree)
    console.print("\n")

    # 5. Final Metrics Summary
    metrics_table = Table(title="[bold green]Final Run Metrics[/bold green]", box=ROUNDED, border_style="green")
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Status / Target", style="dim white")
    metrics_table.add_row("Total Iterations", "1 / 10", "Optimal execution path")
    metrics_table.add_row("Total Token Cost", "1,850", "Budget: 4,000 max")
    metrics_table.add_row("Total Time Elapsed", "1.78s", "Fast response")
    metrics_table.add_row("Resiliency Rate", "100% (1 retry recovered)", "Excellent fault tolerance")
    metrics_table.add_row("Session ID", "research_session_001", "Memory namespace loaded")
    metrics_table.add_row("System Status", "✓ SUCCESS", "Task accomplished successfully")
    
    console.print(metrics_table)
    console.print("\n")

    # 6. Final Structured Output
    json_output = """{
  "status": "success",
  "data": {
    "topic": "quantum_computing_2026",
    "breakthroughs": [
      "Room-temperature qubits validation for scalable architectures",
      "Quantum-assisted molecular simulation for pharmaceutical design"
    ],
    "memory_updated": true,
    "facts_stored": 2
  }
}"""
    
    console.print(
        Panel(
            Syntax(json_output, "json", theme="monokai", line_numbers=True),
            title="[bold yellow]Agent Final Output (Structured JSON)[/bold yellow]",
            border_style="yellow",
            box=ROUNDED,
            expand=False
        )
    )
    console.print("\n")


if __name__ == "__main__":
    main()
