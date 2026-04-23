"""
app.py
Main Streamlit Application — ACity RAG Chatbot
Author: [Your Name] | Index: [Your Index Number]

Features:
- Query input with chat history
- Display retrieved chunks with scores
- Show final prompt sent to LLM
- RAG vs Pure LLM comparison (Part E)
- User feedback (Part G)
- Adversarial query testing (Part E)
"""

import streamlit as st
import os
import sys
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ACity RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --primary: #1a1a2e;
    --accent: #e94560;
    --accent2: #0f3460;
    --surface: #16213e;
    --text: #eaeaea;
    --muted: #8892b0;
    --success: #64ffda;
    --warning: #ffd700;
    --card: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--primary);
    color: var(--text);
}

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #0f1b35 100%);
}

/* Header */
.rag-header {
    background: linear-gradient(90deg, var(--accent2) 0%, var(--accent) 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--accent);
}
.rag-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; color: white; }
.rag-header p { margin: 0.3rem 0 0; color: rgba(255,255,255,0.75); font-size: 0.9rem; }

/* Chat messages */
.chat-user {
    background: var(--accent2);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    border-left: 3px solid var(--accent);
    font-size: 0.95rem;
}
.chat-assistant {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px 16px 16px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Chunks display */
.chunk-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--success);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
}
.score-badge {
    background: var(--accent);
    color: white;
    border-radius: 20px;
    padding: 0.1rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.source-badge {
    background: var(--accent2);
    color: var(--success);
    border-radius: 20px;
    padding: 0.1rem 0.6rem;
    font-size: 0.72rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid var(--border);
}

/* Metrics */
.metric-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--success);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Prompt display */
.prompt-display {
    background: #0a0a14;
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
    color: #c8ffee;
}

/* Warning */
.fail-banner {
    background: rgba(233, 69, 96, 0.15);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: var(--accent);
}

button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
}

.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ─────────────────────────────────────────────
# PIPELINE LOADER
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
LOG_DIR = os.path.join(BASE_DIR, "logs")


@st.cache_resource(show_spinner=False)
def get_pipeline(api_key: str):
    """Build or load the RAG pipeline (cached across reruns)."""
    from rag_pipeline import build_index_from_data, load_pipeline

    if not os.path.exists(os.path.join(INDEX_DIR, "embeddings.npy")):
        with st.spinner("🔧 Building index for the first time... (this takes ~60s)"):
            build_index_from_data(DATA_DIR, INDEX_DIR)

    return load_pipeline(INDEX_DIR, api_key=api_key, log_dir=LOG_DIR)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get yours free at console.anthropic.com"
    )

    st.markdown("---")
    st.markdown("### 🧪 Retrieval Settings")
    top_k = st.slider("Top-K chunks", 3, 10, 5)
    template = st.selectbox(
        "Prompt Template",
        ["v1_structured", "v2_conversational", "v3_chain_of_thought"],
        help="v1=factual, v2=conversational, v3=complex reasoning"
    )
    use_expansion = st.toggle("Query Expansion", value=True,
                               help="Expands query with domain synonyms")

    st.markdown("---")
    st.markdown("### ⚔️ Adversarial Testing")
    adv_mode = st.toggle("Compare: RAG vs Pure LLM", value=False,
                          help="Part E: evidence-based comparison")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    cols = st.columns(2)
    cols[0].metric("Queries", st.session_state.total_queries)
    cols[1].metric("History", len(st.session_state.chat_history))

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.last_trace = None
        st.session_state.total_queries = 0
        if st.session_state.pipeline:
            st.session_state.pipeline.conversation_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Sample Questions:**")
    sample_qs = [
        "Who won the 2020 election in Ashanti Region?",
        "What is Ghana's GDP growth target in the 2025 budget?",
        "Which party had the most votes in Greater Accra in 2016?",
        "What does the 2025 budget say about education?",
        "Compare NPP and NDC votes in Volta Region 2020",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state["prefill_query"] = q


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>🎓 ACity RAG Chatbot</h1>
    <p>Retrieval-Augmented Generation · Ghana Elections 2008–2020 · 2025 Budget Policy · Academic City University</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab_chat, tab_retrieval, tab_prompt, tab_eval, tab_arch, tab_logs = st.tabs([
    "💬 Chat", "🔍 Retrieved Chunks", "📝 Prompt Inspector",
    "⚔️ RAG vs LLM", "🏗️ Architecture", "📋 Logs"
])


# ─────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────
with tab_chat:
    # Display history
    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="chat-user">👤 {turn["query"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-assistant">🤖 {turn["response"]}</div>', unsafe_allow_html=True)

    # Input
    prefill = st.session_state.pop("prefill_query", "")
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.text_input(
                "Ask about Ghana elections or the 2025 budget:",
                value=prefill,
                placeholder="e.g. Who won the Ashanti Region in 2020?",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_query.strip():
        if not api_key:
            st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
        else:
            with st.spinner("🔍 Searching documents and generating answer..."):
                pipeline = get_pipeline(api_key)
                trace = pipeline.run(
                    user_query,
                    top_k=top_k,
                    template=template,
                    use_expansion=use_expansion
                )

            st.session_state.last_trace = trace
            st.session_state.total_queries += 1
            st.session_state.chat_history.append({
                "query": user_query,
                "response": trace["final_response"],
                "trace": trace
            })

            # Show latest response immediately
            st.markdown(f'<div class="chat-user">👤 {user_query}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-assistant">🤖 {trace["final_response"]}</div>', unsafe_allow_html=True)

            # Feedback
            st.markdown("**Rate this answer:**")
            c1, c2, c3, c4, c5 = st.columns(5)
            for rating, col, label in zip([1,2,3,4,5], [c1,c2,c3,c4,c5], ["😞","😕","😐","😊","🤩"]):
                if col.button(f"{label} {rating}", key=f"fb_{st.session_state.total_queries}_{rating}"):
                    pipeline.submit_feedback(user_query, trace["final_response"], rating)
                    st.success(f"Thanks for rating {rating}/5!")

            if trace["stages"]["retrieval"]["retrieval_failed"]:
                st.markdown('<div class="fail-banner">⚠️ Low retrieval confidence — fallback keyword search was used.</div>',
                            unsafe_allow_html=True)

            st.rerun()


# ─────────────────────────────────────────────
# TAB 2: RETRIEVED CHUNKS
# ─────────────────────────────────────────────
with tab_retrieval:
    trace = st.session_state.last_trace
    if trace:
        retrieval = trace["stages"]["retrieval"]
        st.markdown(f"### Retrieved {retrieval['num_results']} chunks for last query")
        st.markdown(f"**Query:** `{trace['query']}`")

        if retrieval.get("retrieval_failed"):
            st.warning("⚠️ Retrieval failed (low scores) — keyword fallback was activated")

        for i, chunk in enumerate(retrieval["results"]):
            score = chunk.get("hybrid_score", 0)
            sim = chunk.get("similarity_score", 0)
            kw = chunk.get("keyword_score", 0)
            source = chunk.get("source", "unknown")

            color = "#64ffda" if score > 0.2 else "#ffd700" if score > 0.05 else "#e94560"
            st.markdown(f"""
            <div class="chunk-card">
                <div style="margin-bottom:0.4rem;">
                    <span class="score-badge" style="background:{color}; color:#000;">
                        #{i+1} score={score:.4f}
                    </span>
                    <span class="source-badge">{source}</span>
                    &nbsp; <span style="color:#8892b0; font-size:0.72rem;">
                        vector={sim:.3f} · keyword={kw:.3f}
                    </span>
                </div>
                <div style="color:#ccd6f6;">{chunk['text'][:400]}{'...' if len(chunk['text'])>400 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Ask a question in the Chat tab to see retrieved chunks here.")


# ─────────────────────────────────────────────
# TAB 3: PROMPT INSPECTOR
# ─────────────────────────────────────────────
with tab_prompt:
    trace = st.session_state.last_trace
    if trace:
        prompt_stage = trace["stages"]["prompt"]
        col1, col2 = st.columns(2)
        col1.metric("Template", prompt_stage["template"])
        col2.metric("Context Size", f"{prompt_stage['context_chars']} chars")

        st.markdown("#### 🔧 System Prompt")
        st.markdown(f'<div class="prompt-display">{prompt_stage["system_prompt"]}</div>',
                    unsafe_allow_html=True)

        st.markdown("#### 📨 Full User Prompt (sent to LLM)")
        st.markdown(f'<div class="prompt-display">{prompt_stage["user_prompt"]}</div>',
                    unsafe_allow_html=True)

        st.markdown("#### 🤖 LLM Response")
        st.markdown(f'<div class="chat-assistant">{trace["stages"]["llm"]["response"]}</div>',
                    unsafe_allow_html=True)

        st.caption(f"⏱️ Total pipeline time: {trace.get('elapsed_seconds', 0)}s")
    else:
        st.info("Ask a question first to inspect the prompt.")


# ─────────────────────────────────────────────
# TAB 4: RAG vs PURE LLM (Part E)
# ─────────────────────────────────────────────
with tab_eval:
    st.markdown("## ⚔️ Critical Evaluation: RAG vs Pure LLM")
    st.markdown("""
    **Part E: Adversarial Testing**
    Compare how RAG (with document retrieval) performs versus a pure LLM with no context.
    """)

    st.markdown("### Adversarial Queries")
    adv_queries = {
        "Ambiguous": "Who won in the north?",
        "Misleading": "Did NPP win every region in 2020?",
        "Out-of-scope": "What is the population of China?",
        "Incomplete": "How much is the allocation?",
    }

    for qtype, q in adv_queries.items():
        st.markdown(f"**{qtype}:** `{q}`")

    st.markdown("---")
    eval_query = st.text_input("Enter a query to compare:", placeholder="Try an adversarial query above")
    if st.button("🔬 Run Comparison", disabled=not (eval_query and api_key)):
        col1, col2 = st.columns(2)
        pipeline = get_pipeline(api_key)

        with col1:
            st.markdown("#### 🔍 RAG Response")
            with st.spinner("Running RAG..."):
                rag_trace = pipeline.run(eval_query, top_k=top_k, template=template)
            st.markdown(f'<div class="chat-assistant">{rag_trace["final_response"]}</div>',
                        unsafe_allow_html=True)
            st.caption(f"Used {rag_trace['stages']['retrieval']['num_results']} chunks")

        with col2:
            st.markdown("#### 🧠 Pure LLM Response (no retrieval)")
            with st.spinner("Running pure LLM..."):
                llm_response = pipeline.run_without_retrieval(eval_query)
            st.markdown(f'<div class="chat-assistant">{llm_response}</div>', unsafe_allow_html=True)
            st.caption("No documents retrieved — general knowledge only")

        st.markdown("#### 📊 Analysis")
        st.info("""
        **Evaluation Criteria:**
        - **Accuracy**: RAG answers should cite specific data from the documents.
        - **Hallucination**: Pure LLM may invent figures; RAG is constrained to context.
        - **Consistency**: Run the same query multiple times; RAG should be more consistent.

        *For quantitative analysis, see the logs tab.*
        """)


# ─────────────────────────────────────────────
# TAB 5: ARCHITECTURE (Part F)
# ─────────────────────────────────────────────
with tab_arch:
    st.markdown("## 🏗️ System Architecture")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                   ACity RAG System                          │
    │                                                             │
    │  INGESTION PIPELINE (run once)                             │
    │  ┌──────────┐   ┌──────────┐   ┌────────────────────┐    │
    │  │  CSV     │   │  PDF     │   │  Data Cleaning     │    │
    │  │ Election │──▶│ Budget   │──▶│  + Chunking        │    │
    │  │  Data    │   │  2025    │   │  (400c / 80c lap)  │    │
    │  └──────────┘   └──────────┘   └────────┬───────────┘    │
    │                                          │                  │
    │                                          ▼                  │
    │                               ┌────────────────────┐       │
    │                               │  TF-IDF Embedder   │       │
    │                               │  (8000-dim vocab)  │       │
    │                               └────────┬───────────┘       │
    │                                        │                    │
    │                                        ▼                    │
    │                               ┌────────────────────┐       │
    │                               │  Numpy VectorStore │       │
    │                               │  (cosine index)    │       │
    │                               └────────────────────┘       │
    │                                                             │
    │  QUERY PIPELINE (per query)                                │
    │  User Query                                                 │
    │      │                                                      │
    │      ▼                                                      │
    │  ┌──────────────┐    ┌─────────────────────────────────┐  │
    │  │ Query        │    │ Hybrid Retriever                 │  │
    │  │ Expansion    │───▶│ (vector + keyword, alpha=0.7)   │  │
    │  └──────────────┘    └──────────────┬──────────────────┘  │
    │                                      │                      │
    │                              Top-K chunks + scores          │
    │                                      │                      │
    │                                      ▼                      │
    │                       ┌─────────────────────────────┐      │
    │                       │  Context Window Manager     │      │
    │                       │  (rank, filter, truncate)   │      │
    │                       └──────────────┬──────────────┘      │
    │                                      │                      │
    │                                      ▼                      │
    │                       ┌─────────────────────────────┐      │
    │                       │  Prompt Builder             │      │
    │                       │  (v1/v2/v3 templates)       │      │
    │                       └──────────────┬──────────────┘      │
    │                                      │                      │
    │                                      ▼                      │
    │                       ┌─────────────────────────────┐      │
    │                       │  Claude claude-sonnet-4      │      │
    │                       │  (via Anthropic API)        │      │
    │                       └──────────────┬──────────────┘      │
    │                                      │                      │
    │                                      ▼                      │
    │                             Final Response                  │
    │                       + Memory update (Part G)              │
    │                       + Feedback logging (Part G)           │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("""
    ### Design Justifications (Part F)

    **Why TF-IDF instead of neural embeddings?**
    - No GPU required; runs on any laptop
    - Interpretable: can inspect which terms drive similarity
    - Sufficient for domain-specific vocabulary (election terms, budget terminology)
    - Trade-off: lower semantic understanding than sentence-transformers, but retrieval
      quality is compensated by hybrid search + query expansion

    **Why hybrid search (vector + keyword)?**
    - Vector search: catches semantic matches (e.g., "who won" → "highest votes")
    - Keyword search: catches exact matches (e.g., exact candidate names, budget figures)
    - alpha=0.7 vector, 0.3 keyword — tuned by testing on 10 sample queries

    **Why chunk size 400 tokens (~1600 chars)?**
    - Budget PDF: policy paragraphs average 350-500 tokens; 400 captures one full idea
    - Election CSV: grouped by region creates 200-600 token blocks; 400 fits most naturally
    - Tested: 200-token chunks lost context; 800-token chunks reduced precision

    **Why Claude API instead of local LLM?**
    - Production quality; no GPU needed
    - Handles long context windows well
    - Easy to constrain with system prompts for hallucination control

    **Part G Innovations:**
    1. **Memory-based RAG**: Conversation history (last 3 turns) is injected into each request
    2. **Feedback loop**: User ratings saved to `logs/feedback.jsonl` for analysis
    3. **Chain-of-thought prompting** (v3 template) for multi-step reasoning
    """)


# ─────────────────────────────────────────────
# TAB 6: LOGS
# ─────────────────────────────────────────────
with tab_logs:
    st.markdown("## 📋 Pipeline Logs")
    st.markdown("Manual experiment logs — all AI pipeline stages recorded automatically.")

    if os.path.exists(LOG_DIR):
        log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.jsonl')]
        if log_files:
            selected_log = st.selectbox("Select log file:", sorted(log_files, reverse=True))
            log_path = os.path.join(LOG_DIR, selected_log)
            with open(log_path) as f:
                lines = f.readlines()
            st.markdown(f"**{len(lines)} log entries**")
            for line in lines[-20:]:  # show last 20
                try:
                    entry = json.loads(line)
                    stage = entry.get("stage", "")
                    ts = entry.get("timestamp", "")[:19]
                    st.markdown(f"`[{ts}] {stage}`")
                    st.json(entry, expanded=False)
                except:
                    pass
        else:
            st.info("No logs yet. Start asking questions!")
    else:
        st.info("Logs directory not created yet.")

    st.markdown("---")
    st.markdown("### 📥 Download Logs")
    if os.path.exists(LOG_DIR):
        for f in os.listdir(LOG_DIR):
            fpath = os.path.join(LOG_DIR, f)
            with open(fpath, 'rb') as fh:
                st.download_button(
                    f"⬇️ {f}",
                    data=fh.read(),
                    file_name=f,
                    key=f"dl_{f}"
                )
