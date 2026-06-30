"""
Streamlit demo UI for the clinical protocol RAG agent.

Two tabs:
  1. Protocol Q&A  — the original 4-agent RAG pipeline, showing retrieved
     context per specialist agent with guardrail status and rerank scores.
  2. Deep Research — the new intent-routing pipeline that decides whether
     to use protocol RAG, ADaM dataset SQL, live web search, or all three,
     then synthesizes a structured markdown report with tables and citations.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agent"))
sys.path.insert(0, str(PROJECT_ROOT / "embeddings"))
sys.path.insert(0, str(PROJECT_ROOT / "ingestion"))

import agentic_pipeline
import deep_research_router as dr
from guardrails import scan_context

st.set_page_config(page_title="Clinical Protocol Research Assistant", layout="wide")

st.title("Clinical Protocol Research Assistant")
st.caption(
    "Multi-agent RAG · ADaM dataset SQL · Live web search · "
    "LangGraph orchestration · OWASP-mapped guardrails"
)


@st.cache_data
def list_protocols() -> list[str]:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    if not processed_dir.exists():
        return []
    return sorted(p.name for p in processed_dir.glob("*.txt"))


# ── Sidebar (shared across both tabs) ────────────────────────────────────────
with st.sidebar:
    st.header("Protocol Filter")
    protocols = list_protocols()
    if not protocols:
        st.warning("No processed protocols found. Run `python ingestion/pdf_extract.py` first.")
    selected = st.selectbox("Limit to one protocol (optional)", ["All protocols"] + protocols)
    protocol_filter = None if selected == "All protocols" else selected

    st.markdown("---")
    st.caption("Active pipeline components:")
    st.markdown(
        "- Section-based chunking (4-level deep)\n"
        "- BGE-large embeddings + ChromaDB\n"
        "- Cross-encoder reranking\n"
        "- OWASP-mapped guardrails (pharma_gxp)\n"
        "- LangGraph multi-agent (parallel fan-out)\n"
        "- Intent router → RAG / SQL / Web\n"
        "- ADaM ADSL + ADAE dataset lookup\n"
        "- Tavily web search"
    )


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_rag, tab_deep = st.tabs(["Protocol Q&A", "Deep Research"])


# ── TAB 1: Protocol Q&A (original pipeline) ──────────────────────────────────
with tab_rag:
    st.subheader("Protocol Q&A")
    st.caption(
        "Runs the full 4-agent LangGraph pipeline (Eligibility / Endpoint / "
        "Design / General → Synthesis) with guardrails-screened retrieval."
    )

    rag_question = st.text_input(
        "Ask a question about the indexed protocols:",
        placeholder="What is the primary efficacy endpoint for the gout trial?",
        key="rag_question",
    )
    rag_clicked = st.button("Ask", type="primary", key="rag_ask")

    if rag_clicked and rag_question:
        with st.spinner("Running multi-agent retrieval and generation..."):
            agent_chunks = {}
            for agent_name, section_titles in agentic_pipeline.AGENT_SECTION_FILTERS.items():
                agent_chunks[agent_name] = agentic_pipeline._retrieve_for_agent(
                    rag_question, section_titles, protocol_filter,
                )
            try:
                answer = agentic_pipeline.ask(rag_question, protocol_filter=protocol_filter)
                error = None
            except Exception as e:
                answer = None
                error = str(e)

        if error:
            st.error(f"Pipeline error: {error}")
        else:
            st.subheader("Answer")
            st.write(answer)

        st.subheader("Retrieved context by specialist agent")
        tabs = st.tabs([name.capitalize() for name in agent_chunks])
        for tab, (agent_name, chunks) in zip(tabs, agent_chunks.items()):
            with tab:
                if not chunks:
                    st.info("No chunks retrieved for this agent's scoped sections.")
                    continue
                rag_chunks = [{"source": c["metadata"]["source_file"], "text": c["text"]} for c in chunks]
                scan_reports = scan_context(rag_chunks, policy="pharma_gxp")
                for chunk, report in zip(chunks, scan_reports):
                    status_color = {"allow": "🟢", "redact": "🟡", "block": "🔴"}[report.action.value]
                    rerank_score = chunk.get("rerank_score")
                    score_label = f"rerank={rerank_score:.3f}" if rerank_score is not None else "rerank=n/a"
                    header = (
                        f"{status_color} {chunk['metadata']['source_file']} | "
                        f"{chunk['metadata']['section_title']} ({score_label})"
                    )
                    with st.expander(header):
                        if report.action.value == "block":
                            st.warning(f"Guardrails BLOCKED: {[str(f) for f in report.findings]}")
                        elif report.action.value == "redact":
                            st.info(f"Guardrails redacted: {[str(f) for f in report.findings]}")
                            st.text(report.text_clean or chunk["text"])
                        else:
                            st.text(chunk["text"])

    elif rag_clicked and not rag_question:
        st.warning("Enter a question first.")
    else:
        st.info("Enter a question above and click **Ask** to run the pipeline.")


# ── TAB 2: Deep Research ─────────────────────────────────────────────────────
with tab_deep:
    st.subheader("Deep Research")
    st.caption(
        "The LLM router decides whether your question needs protocol text (RAG), "
        "trial data (ADaM SQL + PROC SQL display), live web search, or all three — "
        "then synthesizes a structured report with tables and citations."
    )

    dr_question = st.text_input(
        "Ask anything — the router picks the right tools automatically:",
        placeholder="What does the gout protocol say about endpoints, and how many subjects completed the trial?",
        key="dr_question",
    )
    dr_clicked = st.button("Research", type="primary", key="dr_ask")

    if dr_clicked and dr_question:
        with st.spinner("Routing question and running tools in parallel..."):
            try:
                result = dr.deep_research(dr_question, protocol_filter=protocol_filter)
                dr_error = None
            except Exception as e:
                result = None
                dr_error = str(e)

        if dr_error:
            st.error(f"Deep Research error: {dr_error}")
        else:
            # Routing decision metadata
            col1, col2 = st.columns([1, 2])
            with col1:
                tool_labels = {
                    "PROTOCOL_RAG": "📄 Protocol RAG",
                    "DATASET_SQL": "🗄️ Dataset SQL",
                    "WEB_SEARCH": "🌐 Web Search",
                }
                tools_display = " + ".join(
                    tool_labels.get(t, t) for t in result["tools_used"]
                )
                st.metric("Tools invoked", tools_display)
            with col2:
                st.info(f"**Routing reasoning:** {result['routing_reasoning']}")

            # Final report
            st.markdown("---")
            st.markdown(result["report"])

            # Per-tool raw results in expanders
            st.markdown("---")
            st.subheader("Raw tool outputs")
            for tool_result in result["tool_results"]:
                tool = tool_result["tool"]
                label = tool_labels.get(tool, tool)
                with st.expander(f"{label} — raw output"):
                    if not tool_result["success"]:
                        st.error(f"Failed: {tool_result.get('error', 'unknown error')}")
                    elif tool == "DATASET_SQL":
                        r = tool_result["result"]
                        if r.get("sql"):
                            st.code(r["sql"], language="sql")
                        if r.get("proc_sql"):
                            st.caption("PROC SQL equivalent (SAS display only):")
                            st.code(r["proc_sql"], language="sas")
                        if r.get("results"):
                            st.dataframe(r["results"])
                    elif tool == "WEB_SEARCH":
                        for hit in tool_result.get("result", []):
                            st.markdown(f"**[{hit['title']}]({hit['url']})**")
                            st.caption(hit["content"])
                    else:
                        st.write(tool_result.get("result", ""))

    elif dr_clicked and not dr_question:
        st.warning("Enter a question first.")
    else:
        st.info(
            "Enter a question above and click **Research**. "
            "Try something that spans multiple sources, e.g.: "
            "*'What does the gout protocol say about the primary endpoint, "
            "and how many subjects had serious adverse events?'*"
        )
