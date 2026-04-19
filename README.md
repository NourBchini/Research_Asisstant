# Research Assistant

| | |
|---|---|
| **Problem** | Answer a research question with **web-grounded** synthesis—not a single model dump. |
| **Run** | `pip install -r requirements.txt` → copy `.env.example` to `.env` → `python -m research_assistant`. |
| **Result** | A **critic-gated** Markdown brief (sources, risks, next steps) + **Chroma**-indexed follow-up Q&A on that brief. |

A terminal-based **multi-agent research pipeline** built with [Microsoft AutoGen AgentChat](https://github.com/microsoft/autogen), **Groq** (OpenAI-compatible API), **Tavily** (web search and page extract), and **Chroma** with local **SentenceTransformers** embeddings.

You enter a research question; agents search the web, synthesize sources, draft a Markdown brief, review it until approval, then save and index the result so you can ask follow-up questions against that document.

## What it does

1. **Scout** — Finds a few relevant URLs (Tavily search).
2. **Reader** — Reads pages and produces a structured factual synthesis.
3. **Writer** — Turns that into a research brief (Markdown) with sources, risks, and next steps.
4. **Critic** — Scores the brief; the loop ends when output includes `APPROVED`.
5. The brief is **written to disk**, **chunked and embedded** into a local Chroma collection, and you can chat via **ResearchQA** using retrieval over that corpus.

## Requirements

- Python 3.10+ (3.11+ recommended)
- [Groq API key](https://console.groq.com/keys) (LLM calls)
- [Tavily API key](https://tavily.com/) (web search / extract)

Optional: a [Hugging Face token](https://huggingface.co/settings/tokens) as `HF_TOKEN` if you want higher rate limits when the embedding model downloads or updates (not required for basic local use).

## Setup

```bash
git clone https://github.com/NourBchini/Research_Asisstant.git
cd Research_Asisstant

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GROQ_API_KEY and TAVILY_API_KEY
```

## Run

From the repository root (with the venv activated):

```bash
python -m research_assistant
```

At the prompts:

- Enter your **research topic** (or `exit` / empty line to quit).
- After the run, use **You:** for follow-up questions grounded in the indexed brief.
- Choose whether to start a **new** research session when asked.

First launch may download the `all-MiniLM-L6-v2` embedding model (one-time).

## Configuration

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq LLM API ([console](https://console.groq.com/keys)) |
| `TAVILY_API_KEY` | Tavily search/extract |
| `HF_TOKEN` | Optional; embedding / Hub rate limits |

Environment variables can be set in a **`.env`** file at the repo root (loaded automatically by `research_assistant/config.py`).

## Project layout

```
Research_Asisstant/
├── research_assistant/     # Main package
│   ├── main.py             # CLI, session loop, Q&A
│   ├── agents.py           # Scout, Reader, Writer, Critic, ResearchQA
│   ├── tools.py            # Tavily + Chroma VectorDB
│   ├── config.py           # Groq client + env loading
│   ├── data/               # Local Chroma path (created at runtime; gitignored)
│   └── outputs/            # Generated .md briefs (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

Legacy **`Tavily/`** holds an older experiment script; the active app lives under **`research_assistant/`**.

## Notes

- **Secrets:** Never commit `.env`. Only `.env.example` belongs in git.
- **Groq tool calling** on smaller models can occasionally fail; retrying the same topic often works, or you can point Scout/Reader at a larger model in `agents.py`.
- Repository name on GitHub may show as **Research_Asisstant**; you can rename the repo in GitHub settings if you prefer **Research_Assistant**.

