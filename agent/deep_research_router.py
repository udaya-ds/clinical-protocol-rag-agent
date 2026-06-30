"""
Deep Research Router for the clinical protocol RAG agent.

Takes a user question and decides which tools to invoke:
  - PROTOCOL_RAG   → existing 4-agent LangGraph pipeline (protocol text questions)
  - DATASET_SQL    → ADaM dataset lookup via LLM-generated SQL (trial data questions)
  - WEB_SEARCH     → Tavily web search (current events, external evidence)
  - ALL            → all three in parallel, synthesized into one report

Then runs the selected tools, fuses the results, and generates a
structured report with clearly labeled sections and citations.

Architecture follows the diagram in the project brief:

  User Question
       │
       ▼
  Intent Router  (LLM classifies intent → decides tools)
       │
  ┌────┴──────────────────┐──────────────────┐
  ▼                       ▼                  ▼
Protocol RAG         Dataset SQL          Web Search
(agentic_pipeline)  (dataset_lookup)   (web_search_tool)
  │                       │                  │
  └────────────────┬───────────────────────┘
                   ▼
           Result Fusion + Report
           (structured sections,
            tables, citations)
"""

from __future__ import annotations
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / "embeddings"))
sys.path.insert(0, str(PROJECT_ROOT / "ingestion"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

import agentic_pipeline
import dataset_lookup
from web_search_tool import web_search, format_search_results, WebSearchError

client = OpenAI()
MODEL = "gpt-4o"


class ToolSelection(str, Enum):
    PROTOCOL_RAG = "PROTOCOL_RAG"
    DATASET_SQL = "DATASET_SQL"
    WEB_SEARCH = "WEB_SEARCH"
    ALL = "ALL"


ROUTER_SYSTEM_PROMPT = """\
You are an intent router for a clinical trial research assistant. Given a user \
question, decide which tools are needed to answer it. Return ONLY a JSON object \
with exactly two keys: "tools" (a list of one or more tool names) and "reasoning" \
(one sentence explaining why).

Available tools:
- PROTOCOL_RAG: Questions about protocol design, eligibility criteria, endpoints, \
  study design, concomitant medications, safety monitoring - anything that lives in \
  the clinical trial protocol document text.
- DATASET_SQL: Questions about actual trial data - subject counts, demographics, \
  adverse event frequencies, dropout rates, treatment arm statistics. Requires \
  structured ADaM dataset access.
- WEB_SEARCH: Questions requiring current external information - recent regulatory \
  approvals, published trial results, mechanism of action, drug class background, \
  anything not in the protocol or trial dataset.

Rules:
- Use PROTOCOL_RAG for protocol document content questions.
- Use DATASET_SQL for questions about numbers, counts, rates from trial data.
- Use WEB_SEARCH for current/external information not available locally.
- Use multiple tools (or ALL) when the question requires combining sources, \
  e.g. "What does the protocol say about the primary endpoint, and how did the \
  actual efficacy data compare?" needs PROTOCOL_RAG + WEB_SEARCH.
- Return ONLY the JSON object, no markdown, no commentary.

Examples:
- "What are the inclusion criteria?" → {"tools": ["PROTOCOL_RAG"], "reasoning": "..."}
- "How many subjects had serious AEs?" → {"tools": ["DATASET_SQL"], "reasoning": "..."}
- "What is Belimumab's mechanism of action?" → {"tools": ["WEB_SEARCH"], "reasoning": "..."}
- "What does the protocol say about gout flares, and what's the observed AE rate?" \
  → {"tools": ["PROTOCOL_RAG", "DATASET_SQL"], "reasoning": "..."}
"""


def route(question: str) -> tuple[list[ToolSelection], str]:
    """LLM classifies the question → list of tools to invoke + reasoning."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=200,
        temperature=0,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw, strict=False)

    tool_names = data.get("tools", ["PROTOCOL_RAG"])
    reasoning = data.get("reasoning", "")

    # Normalize: if ALL was returned as a single tool name, expand it
    if "ALL" in tool_names:
        tool_names = ["PROTOCOL_RAG", "DATASET_SQL", "WEB_SEARCH"]

    tools = []
    for name in tool_names:
        try:
            tools.append(ToolSelection(name))
        except ValueError:
            pass  # unknown tool name - just skip it

    return tools or [ToolSelection.PROTOCOL_RAG], reasoning


def _run_rag(question: str, protocol_filter: str | None) -> dict:
    try:
        answer = agentic_pipeline.ask(question, protocol_filter=protocol_filter)
        return {"tool": "PROTOCOL_RAG", "success": True, "result": answer}
    except Exception as e:
        return {"tool": "PROTOCOL_RAG", "success": False, "error": str(e)}


def _run_sql(question: str) -> dict:
    try:
        result = dataset_lookup.answer_dataset_question(question)
        return {"tool": "DATASET_SQL", "success": True, "result": result}
    except Exception as e:
        return {"tool": "DATASET_SQL", "success": False, "error": str(e)}


def _run_web(question: str) -> dict:
    try:
        results = web_search(question)
        return {
            "tool": "WEB_SEARCH",
            "success": True,
            "result": results,
            "formatted": format_search_results(results),
        }
    except WebSearchError as e:
        return {"tool": "WEB_SEARCH", "success": False, "error": str(e)}


REPORT_SYSTEM_PROMPT = """\
You are a clinical research analyst generating a structured research report. You \
will receive findings from one or more data sources (protocol documents, ADaM \
trial datasets, web search) and must synthesize them into a clear, well-organized \
answer with the following structure:

## Summary
One paragraph directly answering the question.

## Findings
Subsections per data source used (e.g., ### From Protocol, ### From Trial Data, \
### From Web Search). Include specific details, numbers, and direct citations. \
For trial data, present counts/rates in a markdown table where appropriate.

## Key Takeaways
2-4 bullet points of the most important, actionable conclusions.

Rules:
- Always cite sources (protocol section, dataset variable, or URL).
- Never fabricate data or citations not present in the provided findings.
- If a data source returned an error or no results, note it briefly.
- Keep the tone professional and clinical-analyst appropriate.
"""


def generate_report(
    question: str,
    tool_results: list[dict],
    routing_reasoning: str,
) -> str:
    """Fuse all tool results into one structured markdown report."""
    sections = []
    sections.append(f"**Question:** {question}")
    sections.append(f"**Routing decision:** {routing_reasoning}")
    sections.append("")

    for result in tool_results:
        tool = result["tool"]
        if not result["success"]:
            sections.append(f"**{tool}:** Failed - {result.get('error', 'unknown error')}")
            continue

        if tool == "PROTOCOL_RAG":
            sections.append(f"**Protocol RAG findings:**\n{result['result']}")

        elif tool == "DATASET_SQL":
            r = result["result"]
            if r.get("blocked"):
                sections.append(f"**Dataset SQL:** Query was blocked - {r['block_reason']}")
            elif r.get("error"):
                sections.append(f"**Dataset SQL:** Query error - {r['error']}")
            else:
                rows = r.get("results", [])
                sql = r.get("sql", "")
                proc_sql = r.get("proc_sql", "")
                sections.append(
                    f"**Dataset SQL findings:**\n"
                    f"SQL executed: `{sql}`\n\n"
                    f"PROC SQL equivalent (SAS display):\n```sas\n{proc_sql}\n```\n\n"
                    f"Results ({len(rows)} rows): {json.dumps(rows[:20], indent=2)}"
                )

        elif tool == "WEB_SEARCH":
            sections.append(f"**Web search findings:**\n{result.get('formatted', '')}")

    combined_findings = "\n\n".join(sections)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        temperature=0,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": combined_findings},
        ],
    )
    return response.choices[0].message.content


def deep_research(
    question: str,
    protocol_filter: str | None = None,
    max_workers: int = 3,
) -> dict:
    """Full pipeline: route → run tools in parallel → generate report.

    Returns a dict with:
      - question
      - tools_used (list of tool names)
      - routing_reasoning
      - tool_results (raw per-tool outputs)
      - report (final structured markdown report)
    """
    tools, reasoning = route(question)
    tool_names = [t.value for t in tools]

    # Run selected tools in parallel - same True Parallel fan-out pattern as
    # agentic_pipeline.py's specialist agents; tools don't depend on each
    # other's outputs, so there's no reason to run them sequentially.
    tool_results = []
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if ToolSelection.PROTOCOL_RAG in tools:
            futures[executor.submit(_run_rag, question, protocol_filter)] = "PROTOCOL_RAG"
        if ToolSelection.DATASET_SQL in tools:
            futures[executor.submit(_run_sql, question)] = "DATASET_SQL"
        if ToolSelection.WEB_SEARCH in tools:
            futures[executor.submit(_run_web, question)] = "WEB_SEARCH"

        for future in as_completed(futures):
            tool_results.append(future.result())

    # Sort by tool name for deterministic output ordering
    tool_results.sort(key=lambda r: r["tool"])

    report = generate_report(question, tool_results, reasoning)

    return {
        "question": question,
        "protocol_filter": protocol_filter or "all",
        "tools_used": tool_names,
        "routing_reasoning": reasoning,
        "tool_results": tool_results,
        "report": report,
    }


if __name__ == "__main__":
    question = input("Deep Research question: ").strip()
    if not question:
        sys.exit(0)

    filter_input = input(
        "Optional protocol filter (press Enter for all): "
    ).strip() or None

    print("\nRouting and running tools...\n")
    result = deep_research(question, protocol_filter=filter_input)

    print(f"Tools used: {result['tools_used']}")
    print(f"Routing reasoning: {result['routing_reasoning']}")
    print("\n" + "=" * 70)
    print(result["report"])
