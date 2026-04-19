import asyncio
from datetime import datetime
from pathlib import Path

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule

from research_assistant.agents import get_research_qa_agent, get_research_team
from research_assistant.config import get_model
from research_assistant.tools import VectorDB

_PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _PACKAGE_ROOT / "outputs"

console = Console()


async def run_research_session(task: str, db: VectorDB) -> str:
    scout, reader, writer, critic = get_research_team()

    team = SelectorGroupChat(
        participants=[scout, reader, writer, critic],
        model_client=get_model("llama-3.3-70b-versatile"),
        termination_condition=TextMentionTermination("APPROVED"),
        selector_prompt=(
            "You orchestrate a research-assistant team. "
            "Available agents:\n"
            "- Scout: finds relevant URLs on the web\n"
            "- Reader: reads URLs and produces a structured factual synthesis. "
            "If the Writer asks something with [INFO_NEEDED:...], the Reader must answer.\n"
            "- Writer: drafts the Markdown brief from the synthesis. "
            "May use [INFO_NEEDED: <point>] to ask the Reader for more detail.\n"
            "- Critic: reviews the brief (sources, rigor, structure). Writes 'APPROVED' if it is good enough, "
            "otherwise requests revisions for the Writer.\n\n"
            "Conversation history:\n{history}\n\n"
            "Which agent should speak next? Reply with only the agent name."
        ),
    )

    research_brief = ""
    async for message in team.run_stream(task=task):
        source = getattr(message, "source", "")
        content = getattr(message, "content", "")
        if not content or not isinstance(content, str):
            continue

        colors = {"Scout": "blue", "Reader": "yellow", "Writer": "green", "Critic": "red"}
        color = colors.get(source, "dim")
        preview = content[:300] + ("..." if len(content) > 300 else "")
        console.print(f"[{color}][{source}][/{color}] [dim]{preview}[/dim]")

        if source == "Writer":
            research_brief = content

    return research_brief


async def main():
    console.print(Rule("[bold cyan]Research assistant[/bold cyan]"))
    console.print(
        "[dim]Type your research question at the prompt below. "
        "Use [bold]exit[/bold] / empty line to leave. "
        "Run in a real terminal ([bold]python -m research_assistant[/bold]) so typing works.[/dim]\n"
    )

    while True:
        topic = Prompt.ask(
            "[bold cyan]Research topic / question[/bold cyan]",
            console=console,
            default="",
            show_default=False,
        ).strip()

        if topic.lower() in ("exit", "quit", ""):
            console.print("[dim]Goodbye.[/dim]")
            return

        db = VectorDB(reset=True)

        console.print(Panel(topic, title="[bold]Session[/bold]", border_style="cyan"))
        research_brief = await run_research_session(topic, db)

        if not research_brief or len(research_brief) < 400:
            console.print("[red]Brief too short or empty. Try another question or exit.[/red]\n")
            if not Confirm.ask("Start another session?", default=True, console=console):
                console.print("[dim]Goodbye.[/dim]")
                return
            continue

        console.print("[dim]→ Indexing into the vector store...[/dim]")
        console.print(f"[dim]{db.index_segments(research_brief)}[/dim]")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = OUTPUT_DIR / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filename.write_text(research_brief, encoding="utf-8")

        console.print(Rule())
        console.print(
            f"[bold green]Brief saved → [cyan]{filename}[/cyan] ({len(research_brief)} characters)[/bold green]"
        )

        console.print(Rule("[bold magenta]Follow-up questions (indexed corpus)[/bold magenta]"))
        console.print("[dim]Chat using the indexed brief. [bold]exit[/bold] / empty line ends Q&A.[/dim]\n")

        qa_agent = get_research_qa_agent(db)
        qa_team = RoundRobinGroupChat([qa_agent], termination_condition=MaxMessageTermination(2))

        while True:
            question = Prompt.ask("[bold]You[/bold]", console=console, default="", show_default=False).strip()
            if question.lower() in ("exit", "quit", ""):
                break
            result = await qa_team.run(task=question)
            for msg in result.messages:
                if msg.source == "ResearchQA":
                    console.print(
                        Panel(
                            msg.content,
                            title="[bold magenta]Assistant[/bold magenta]",
                            border_style="magenta",
                        )
                    )

        console.print()
        if not Confirm.ask("Start a [bold]new[/bold] research session?", default=False, console=console):
            console.print("[dim]Goodbye.[/dim]")
            return


if __name__ == "__main__":
    asyncio.run(main())
