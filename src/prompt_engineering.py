"""
prompt_engineering.py
Part C: Prompt Engineering & Generation
Author: [Your Name] | Index: [Your Index Number]

Implements:
- Context-injecting prompt templates with hallucination control
- Context window management (ranking + truncation)
- Multiple prompt variants for experimentation
"""

import json
from datetime import datetime


# ─────────────────────────────────────────────
# CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────

MAX_CONTEXT_CHARS = 6000  # ~1500 tokens reserved for retrieved context

def rank_and_filter_chunks(chunks: list[dict], query: str,
                            max_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
    """
    Rank chunks by hybrid score, then truncate to fit context window.
    Also filters chunks with score below minimum threshold.
    """
    MIN_SCORE = 0.01  # filter near-zero relevance chunks
    filtered = [c for c in chunks if c.get('hybrid_score', 0) >= MIN_SCORE]
    filtered.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

    selected = []
    total_chars = 0
    for chunk in filtered:
        chunk_len = len(chunk['text'])
        if total_chars + chunk_len <= max_chars:
            selected.append(chunk)
            total_chars += chunk_len
        else:
            # Truncate the last chunk to fit
            remaining = max_chars - total_chars
            if remaining > 200:
                truncated = chunk.copy()
                truncated['text'] = chunk['text'][:remaining] + "..."
                selected.append(truncated)
            break

    return selected


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = chunk.get('source', 'unknown')
        score = chunk.get('hybrid_score', 0)
        parts.append(
            f"[Context {i}] (Source: {source_label} | Relevance: {score:.3f})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# Three variants tested experimentally
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are ACity RAG Assistant, an AI assistant for Academic City University.
You answer questions about Ghana's 2025 Budget Policy and Ghana Presidential Election Results.

STRICT RULES:
1. Only answer using the provided CONTEXT. Do NOT use outside knowledge.
2. If the context does not contain the answer, say: "I don't have enough information in the provided documents to answer that."
3. Cite which Context number your answer comes from (e.g., "According to Context 2...").
4. For numbers and statistics, quote them exactly from the context.
5. Never guess, estimate, or fabricate data.
"""

# ── TEMPLATE 1: Structured (best for factual queries) ──
def build_prompt_v1(query: str, context_text: str) -> str:
    """
    Template V1: Structured with explicit hallucination guard.
    Best for: Factual election data, budget figures.
    Experiment result: Highest precision on numerical queries.
    """
    return f"""CONTEXT DOCUMENTS:
{context_text}

---
USER QUESTION: {query}

INSTRUCTIONS:
- Answer ONLY using the CONTEXT above.
- If the answer is in the context, provide it clearly with the citation.
- If NOT found, reply: "The provided documents do not contain information about this."
- Keep your answer concise and factual.

ANSWER:"""


# ── TEMPLATE 2: Conversational (better UX, slightly lower precision) ──
def build_prompt_v2(query: str, context_text: str) -> str:
    """
    Template V2: Conversational style.
    Best for: Explanatory questions about budget policy.
    Experiment result: Better readability but occasionally paraphrases loosely.
    """
    return f"""You are answering as ACity RAG Assistant. Use ONLY the context below.

Context:
{context_text}

Question: {query}

Answer in a clear, helpful tone. If the context lacks the answer, say so honestly. 
Do not add information not in the context."""


# ── TEMPLATE 3: Chain-of-thought (Part G: Innovation — multi-step reasoning) ──
def build_prompt_v3(query: str, context_text: str) -> str:
    """
    Template V3: Chain-of-thought for complex reasoning queries.
    Best for: Comparative questions, multi-step reasoning.
    Experiment result: More accurate on ambiguous queries; slower.
    """
    return f"""CONTEXT:
{context_text}

QUESTION: {query}

Think step by step:
1. Identify which part(s) of the CONTEXT are relevant to the question.
2. Extract the key facts.
3. Reason through the answer.
4. State your final answer clearly, citing Context numbers.

STEP-BY-STEP REASONING:"""


PROMPT_VARIANTS = {
    "v1_structured": build_prompt_v1,
    "v2_conversational": build_prompt_v2,
    "v3_chain_of_thought": build_prompt_v3,
}


# ─────────────────────────────────────────────
# PROMPT BUILDER (main interface)
# ─────────────────────────────────────────────

def build_final_prompt(query: str, retrieved_chunks: list[dict],
                        template: str = "v1_structured",
                        log: bool = True) -> dict:
    """
    Main function: selects context, builds prompt, returns everything for logging.
    """
    # 1. Filter and rank context
    selected_chunks = rank_and_filter_chunks(retrieved_chunks, query)

    # 2. Format context block
    context_text = format_context(selected_chunks)

    # 3. Choose template
    builder = PROMPT_VARIANTS.get(template, build_prompt_v1)
    user_prompt = builder(query, context_text)

    result = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "template_used": template,
        "selected_chunks": len(selected_chunks),
        "context_chars": len(context_text),
        "context_text": context_text,
        "timestamp": datetime.now().isoformat()
    }

    if log:
        print(f"\n[PromptBuilder] Template: {template}")
        print(f"[PromptBuilder] Context: {len(selected_chunks)} chunks, {len(context_text)} chars")
        print(f"[PromptBuilder] Prompt preview:\n{user_prompt[:300]}...")

    return result


# ─────────────────────────────────────────────
# EXPERIMENT LOGGER
# ─────────────────────────────────────────────

def log_prompt_experiment(query: str, template: str, prompt: str,
                           response: str, log_file: str):
    """Save prompt experiment for analysis."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "template": template,
        "prompt_length": len(prompt),
        "response_preview": response[:200],
        "response_length": len(response),
    }
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[Logger] Could not write experiment log: {e}")
