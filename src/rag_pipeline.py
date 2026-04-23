"""
rag_pipeline.py
Part D: Full RAG Pipeline Implementation
Author: [Your Name] | Index: [Your Index Number]

Pipeline: User Query → Retrieval → Context Selection → Prompt → LLM → Response
Includes: Full logging at each stage, displayed retrieved docs, scores, final prompt.
"""

import os
import json
import time
from datetime import datetime
import anthropic

# ─────────────────────────────────────────────
# PIPELINE LOGGER
# ─────────────────────────────────────────────

class PipelineLogger:
    """Logs every step of the RAG pipeline to file + console."""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"pipeline_{self.session_id}.jsonl")
        self.query_count = 0

    def log(self, stage: str, data: dict):
        entry = {
            "session": self.session_id,
            "query_num": self.query_count,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [LOG:{stage}] Recorded to {os.path.basename(self.log_file)}")
        return entry


# ─────────────────────────────────────────────
# FULL RAG PIPELINE
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    Complete RAG pipeline with stage-by-stage logging.
    Exposes full trace for UI display and experiment logs.
    """

    def __init__(self, retriever, api_key: str = None, log_dir: str = "logs"):
        self.retriever = retriever
        self.logger = PipelineLogger(log_dir=log_dir)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.conversation_history = []  # Part G: Memory-based RAG
        self.query_feedback = {}        # Part G: Feedback loop

        # Import here to keep things modular
        from prompt_engineering import build_final_prompt, log_prompt_experiment
        self.build_final_prompt = build_final_prompt
        self.log_prompt_experiment = log_prompt_experiment

    def run(self, query: str, top_k: int = 5,
            template: str = "v1_structured",
            use_expansion: bool = True) -> dict:
        """
        Execute the full pipeline for one query.
        Returns a full trace dict for UI display.
        """
        self.logger.query_count += 1
        trace = {"query": query, "stages": {}}
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"[RAG Pipeline] Query #{self.logger.query_count}: {query}")
        print(f"{'='*60}")

        # ── STAGE 1: Query received ──────────────────────
        self.logger.log("1_query_received", {"query": query, "top_k": top_k,
                                              "template": template})
        trace["stages"]["query"] = {"query": query}

        # ── STAGE 2: Retrieval ───────────────────────────
        print("\n[STAGE 2] Retrieval...")
        retrieval_result = self.retriever.retrieve(
            query, top_k=top_k, use_expansion=use_expansion, log_steps=True
        )
        retrieved = retrieval_result["results"]
        retrieval_log = retrieval_result["log"]

        self.logger.log("2_retrieval", {
            "num_results": len(retrieved),
            "top_score": retrieved[0]["hybrid_score"] if retrieved else 0,
            "retrieval_failed": retrieval_result["retrieval_failed"],
            "results_preview": [
                {"chunk_id": r["chunk_id"], "score": r["hybrid_score"],
                 "text_50": r["text"][:50]}
                for r in retrieved
            ]
        })
        trace["stages"]["retrieval"] = {
            "num_results": len(retrieved),
            "retrieval_failed": retrieval_result["retrieval_failed"],
            "results": retrieved
        }

        # ── STAGE 3: Prompt Construction ────────────────
        print("\n[STAGE 3] Building prompt...")
        prompt_data = self.build_final_prompt(query, retrieved, template=template, log=True)
        self.logger.log("3_prompt_built", {
            "template": template,
            "selected_chunks": prompt_data["selected_chunks"],
            "context_chars": prompt_data["context_chars"],
            "prompt_preview": prompt_data["user_prompt"][:300]
        })
        trace["stages"]["prompt"] = {
            "template": template,
            "system_prompt": prompt_data["system_prompt"],
            "user_prompt": prompt_data["user_prompt"],
            "context_text": prompt_data["context_text"],
            "context_chars": prompt_data["context_chars"]
        }

        # ── STAGE 4: LLM Generation ─────────────────────
        print("\n[STAGE 4] Calling Claude API...")
        llm_response = ""
        error_msg = ""

        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            # Part G: Memory — include conversation history for context continuity
            messages = list(self.conversation_history[-6:])  # keep last 3 turns
            messages.append({"role": "user", "content": prompt_data["user_prompt"]})

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=prompt_data["system_prompt"],
                messages=messages
            )
            llm_response = response.content[0].text
            print(f"[STAGE 4] Response received ({len(llm_response)} chars)")

            # Update conversation memory
            self.conversation_history.append(
                {"role": "user", "content": query}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": llm_response}
            )

        except Exception as e:
            error_msg = str(e)
            llm_response = f"⚠ API Error: {error_msg}\n\nPlease check your ANTHROPIC_API_KEY."
            print(f"[STAGE 4] ERROR: {error_msg}")

        elapsed = time.time() - start_time
        self.logger.log("4_llm_response", {
            "response_chars": len(llm_response),
            "elapsed_seconds": round(elapsed, 2),
            "error": error_msg,
            "response_preview": llm_response[:200]
        })

        trace["stages"]["llm"] = {
            "response": llm_response,
            "elapsed": round(elapsed, 2),
            "error": error_msg
        }

        # ── STAGE 5: Log experiment ──────────────────────
        exp_log = os.path.join(self.logger.log_dir, "prompt_experiments.jsonl")
        self.log_prompt_experiment(query, template, prompt_data["user_prompt"],
                                    llm_response, exp_log)

        trace["final_response"] = llm_response
        trace["elapsed_seconds"] = round(elapsed, 2)

        print(f"\n[RAG Pipeline] Done in {elapsed:.2f}s")
        print(f"{'='*60}\n")

        return trace

    def run_without_retrieval(self, query: str) -> str:
        """
        Part E: Pure LLM baseline (no RAG) for comparison.
        """
        print(f"\n[BASELINE] Running pure LLM (no retrieval): {query}")
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system="You are a helpful assistant. Answer based on your general knowledge.",
                messages=[{"role": "user", "content": query}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {e}"

    def submit_feedback(self, query: str, response: str, rating: int, comment: str = ""):
        """
        Part G: Feedback loop.
        Stores user feedback to improve retrieval in future sessions.
        """
        feedback_file = os.path.join(self.logger.log_dir, "feedback.jsonl")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "rating": rating,  # 1-5
            "comment": comment,
            "response_preview": response[:100]
        }
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

        self.query_feedback[query] = rating
        print(f"[Feedback] Saved rating {rating} for query: '{query[:50]}'")
        return entry


# ─────────────────────────────────────────────
# INDEX BUILDER (run once to create index)
# ─────────────────────────────────────────────

def build_index_from_data(data_dir: str, index_dir: str):
    """Build and save the RAG index. Run this once before starting the app."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_ingestion import ingest_all_data
    from retrieval import VectorStore, HybridRetriever

    print("[Setup] Ingesting data...")
    chunks = ingest_all_data(data_dir)

    print("[Setup] Building vector index...")
    vs = VectorStore()
    vs.build_index(chunks)
    vs.save(index_dir)

    print(f"[Setup] Index saved to {index_dir}")
    return len(chunks)


def load_pipeline(index_dir: str, api_key: str, log_dir: str = "logs") -> RAGPipeline:
    """Load existing index and create pipeline."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retrieval import VectorStore, HybridRetriever

    vs = VectorStore()
    vs.load(index_dir)
    retriever = HybridRetriever(vs)
    pipeline = RAGPipeline(retriever, api_key=api_key, log_dir=log_dir)
    print("[Pipeline] Ready.")
    return pipeline
