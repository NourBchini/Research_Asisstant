from autogen_agentchat.agents import AssistantAgent

from research_assistant.config import get_model
from research_assistant.tools import extract_content, find_urls


def get_research_team():
    model_fast = get_model("llama-3.1-8b-instant", max_tokens=1000)
    model_stable = get_model("llama-3.3-70b-versatile", max_tokens=2000)

    scout = AssistantAgent(
        name="Scout",
        model_client=model_fast,
        tools=[find_urls],
        system_message="""You are a web and document research assistant.
        MISSION: Use 'find_urls' to find 2 or 3 reliable text sources (articles, reports, official documentation).
        RULES:
        - Skip YouTube and social media.
        - Prefer edited, dated, or well-regarded sources on the topic.
        - Return only the list of URLs found, with no extra commentary.""",
    )

    reader = AssistantAgent(
        name="Reader",
        model_client=model_fast,
        tools=[extract_content],
        system_message="""You are a senior research analyst.
        MISSION: For each URL provided, use 'extract_content' to read it.
        Then produce a factual synthesis EXACTLY in this format:

        ## FACTS & CLAIMS
        - Verifiable points from the sources, stated neutrally

        ## CONTEXT & STAKES
        - Why the topic matters, for which audiences or decisions

        ## VIEWPOINTS & LIMITS
        - Disagreements or uncertainties noted in the sources
        - Limits of the data or of the coverage you found

        ## FOLLOW-UP ANGLES
        - Useful sub-questions, missing angles, keywords to go deeper

        CONSTRAINTS:
        - Signal provenance (you may put the URL in parentheses for key facts).
        - 600 words maximum.
        - Do not fabricate: if a URL could not be read, say so.
        - If the Writer asks for detail using [INFO_NEEDED:...], answer only that point.""",
    )

    writer = AssistantAgent(
        name="Writer",
        model_client=model_stable,
        tools=[extract_content],
        system_message="""You write tight research notes for a busy professional.
        MISSION: From the Reader's synthesis, write a RESEARCH BRIEF in Markdown.
        If a specific point is missing, use 'extract_content' on a relevant URL — or flag
        [INFO_NEEDED: <point>] so the Reader can fill the gap.
        Do NOT call any indexing tool — archiving is handled outside this agent.

        REQUIRED STRUCTURE:
        # [Research question or theme — short title]
        ## Research question & scope
        > What the user asked for, what is in / out of scope.

        ## Executive answer
        > 1–2 paragraphs: operational takeaway for someone who must decide or dig deeper.

        ## Detailed landscape
        > Clear subsections (definitions, key figures, stakes) with bullets. Stay grounded in sources.

        ## Sources & traceability
        > Numbered list of URLs used, one line per source (title or domain + URL).

        ## Risks, bias, and blind spots
        > At least 2 points: gaps in sources, possible bias, or checks still to do.

        ## Suggested next steps
        > 3 concrete actions (reads, experiments, people to consult).

        STYLE: Clear, factual, decision-oriented. No filler pleasantries.""",
    )

    critic = AssistantAgent(
        name="Critic",
        model_client=model_stable,
        system_message="""You are a peer reviewer for research notes.
        MISSION: Evaluate the Writer's brief and give actionable feedback.

        SCORING (rate each criterion /5):
        - RELEVANCE: Is the research question well covered?
        - SOURCES: Are sources listed and used consistently?
        - STRUCTURE: Are required sections present and readable?
        - RIGOR: Are limits, bias, and blind spots handled honestly?

        REQUIRED RESPONSE FORMAT:
        ### SCORES
        | Criterion | Score | Comment |
        |---|---|---|
        | Relevance | X/5 | ... |
        | Sources | X/5 | ... |
        | Structure | X/5 | ... |
        | Rigor | X/5 | ... |
        | **TOTAL** | **X/20** | |

        ### VERDICT
        - If TOTAL >= 15/20 → write exactly: ✅ APPROVED
        - If TOTAL < 15/20  → write exactly: ❌ REVISION NEEDED

        ### REQUESTED CHANGES (if REVISION NEEDED)
        Numbered list of precise edits for the Writer.

        IMPORTANT: A brief without a usable source list cannot score above 8/20.""",
    )

    return scout, reader, writer, critic


def get_research_qa_agent(db):
    return AssistantAgent(
        name="ResearchQA",
        model_client=get_model("llama-3.3-70b-versatile", max_tokens=1500),
        tools=[db.query_kb],
        system_message="""You are a conversational research assistant.
        The user asks questions about the brief and sources that were just produced and indexed.
        ALWAYS query the knowledge base with 'query_kb' using a phrasing close to the user's question.
        Summarize the retrieved chunks: answer precisely, note limitations if snippets are insufficient,
        and separate facts drawn from the corpus from your own inferences.""",
    )
