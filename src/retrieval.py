"""
retrieval.py
Part B: Custom Retrieval System
Author: [Your Name] | Index: [Your Index Number]

Implements:
- Custom embedding pipeline (TF-IDF + cosine similarity — no external embedding API needed)
- Vector storage using numpy arrays (no FAISS dependency for portability)
- Top-k retrieval with similarity scoring
- Hybrid search: keyword (BM25-style TF-IDF) + vector (cosine) combination
- Query expansion for improved recall
- Failure case detection and fallback
"""

import numpy as np
import json
import os
import re
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime


# ─────────────────────────────────────────────
# EMBEDDING PIPELINE
# Uses TF-IDF vectorization — no API required, fully reproducible
# ─────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Custom TF-IDF embedder.
    Chosen because:
    - No external API or GPU needed
    - Deterministic and explainable
    - Works well for domain-specific text (election data, budget policy)
    - Can be extended with domain vocabulary boosts
    """

    def __init__(self, max_features: int = 8000):
        self.max_features = max_features
        self.vocabulary = {}        # term -> index
        self.idf = {}              # term -> idf score
        self.is_fitted = False

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Basic stopword removal
        stopwords = {
            'the','a','an','is','are','was','were','be','been','being',
            'have','has','had','do','does','did','will','would','could',
            'should','may','might','shall','can','to','of','in','for',
            'on','with','at','by','from','as','into','through','about',
            'and','or','but','if','its','it','this','that','these','those',
            'we','you','he','she','they','them','their','our','your','his','her'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def fit(self, texts: list[str]):
        """Build vocabulary and IDF from corpus."""
        print("[Embedder] Building TF-IDF vocabulary...")
        N = len(texts)
        df = defaultdict(int)  # document frequency

        tokenized = []
        for text in texts:
            tokens = self._tokenize(text)
            tokenized.append(tokens)
            for term in set(tokens):
                df[term] += 1

        # Select top-N terms by document frequency
        sorted_terms = sorted(df.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}

        # Compute IDF: log((N + 1) / (df + 1)) + 1  (smoothed)
        self.idf = {}
        for term, idx in self.vocabulary.items():
            self.idf[term] = math.log((N + 1) / (df[term] + 1)) + 1

        self.is_fitted = True
        print(f"[Embedder] Vocabulary size: {len(self.vocabulary)}")
        return tokenized

    def transform(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        for term, count in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                tf_score = count / total
                vec[idx] = tf_score * self.idf.get(term, 1.0)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        print("[Embedder] Transforming corpus...")
        matrix = np.array([self.transform(t) for t in texts], dtype=np.float32)
        print(f"[Embedder] Embedding matrix shape: {matrix.shape}")
        return matrix

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'vocabulary': self.vocabulary, 'idf': self.idf,
                         'max_features': self.max_features, 'is_fitted': self.is_fitted}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocabulary = data['vocabulary']
        self.idf = data['idf']
        self.max_features = data['max_features']
        self.is_fitted = data['is_fitted']


# ─────────────────────────────────────────────
# VECTOR STORE (numpy-based)
# ─────────────────────────────────────────────

class VectorStore:
    """
    Custom numpy-based vector store.
    Stores embeddings + metadata. Supports cosine similarity search.
    """

    def __init__(self):
        self.embeddings = None      # np.ndarray shape [N, D]
        self.chunks = []            # list of chunk dicts
        self.embedder = TFIDFEmbedder()

    def build_index(self, chunks: list[dict]):
        """Embed all chunks and store."""
        print(f"[VectorStore] Building index for {len(chunks)} chunks...")
        texts = [c['text'] for c in chunks]
        self.chunks = chunks
        self.embeddings = self.embedder.fit_transform(texts)
        print(f"[VectorStore] Index built. Shape: {self.embeddings.shape}")

    def cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all stored vectors."""
        # Embeddings are already L2-normalized, so dot product = cosine similarity
        return self.embeddings @ query_vec

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Top-k retrieval with similarity scores."""
        query_vec = self.embedder.transform(query)
        scores = self.cosine_similarity(query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['similarity_score'] = float(scores[idx])
            results.append(chunk)

        return results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(index_dir, "chunks.json"), 'w') as f:
            json.dump(self.chunks, f)
        self.embedder.save(os.path.join(index_dir, "embedder.pkl"))
        print(f"[VectorStore] Index saved to {index_dir}")

    def load(self, index_dir: str):
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "chunks.json"), 'r') as f:
            self.chunks = json.load(f)
        self.embedder.load(os.path.join(index_dir, "embedder.pkl"))
        print(f"[VectorStore] Loaded index: {len(self.chunks)} chunks")


# ─────────────────────────────────────────────
# QUERY EXPANSION (Part B extension)
# ─────────────────────────────────────────────

EXPANSION_MAP = {
    # Election domain
    "election": ["vote", "presidential", "candidate", "result", "party"],
    "npp": ["new patriotic party", "akufo addo", "bawumia"],
    "ndc": ["national democratic congress", "mahama", "atta mills"],
    "vote": ["election", "ballot", "result", "candidate", "votes"],
    "region": ["constituency", "district", "area", "zone"],
    "winner": ["most votes", "highest votes", "leading candidate"],
    "loser": ["least votes", "lowest votes", "second place"],
    "2020": ["2020 election", "presidential 2020"],
    "2016": ["2016 election", "presidential 2016"],
    # Budget domain
    "budget": ["fiscal", "expenditure", "revenue", "policy", "economic"],
    "gdp": ["gross domestic product", "economic growth", "economy"],
    "tax": ["taxation", "revenue", "levy", "duty", "fiscal"],
    "inflation": ["price level", "cost of living", "cpi"],
    "debt": ["borrowing", "fiscal deficit", "loans", "liabilities"],
    "education": ["school", "students", "learning", "academic"],
    "health": ["hospital", "medical", "healthcare", "nhis"],
    "infrastructure": ["roads", "bridges", "construction", "development"],
}

def expand_query(query: str) -> str:
    """
    Expand query with domain synonyms to improve recall.
    Strategy: append related terms without replacing original query.
    This prevents query drift while improving recall.
    """
    query_lower = query.lower()
    expansions = []
    for key, synonyms in EXPANSION_MAP.items():
        if key in query_lower:
            expansions.extend(synonyms)

    if expansions:
        expanded = query + " " + " ".join(set(expansions))
        print(f"[QueryExpansion] Original: '{query}'")
        print(f"[QueryExpansion] Expanded: '{expanded[:100]}...'")
        return expanded
    return query


# ─────────────────────────────────────────────
# HYBRID RETRIEVAL SYSTEM
# Combines: TF-IDF vector search + keyword (BM25-style) scoring
# ─────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid search = vector similarity + keyword overlap.
    Final score = alpha * vector_score + (1 - alpha) * keyword_score
    """

    def __init__(self, vector_store: VectorStore, alpha: float = 0.7):
        self.vs = vector_store
        self.alpha = alpha  # weight for vector search; 1-alpha for keyword

    def keyword_score(self, query: str, text: str) -> float:
        """Simple keyword overlap score (BM25-inspired)."""
        query_terms = set(query.lower().split())
        text_lower = text.lower()
        hits = sum(1 for term in query_terms if term in text_lower)
        return hits / max(len(query_terms), 1)

    def retrieve(self, query: str, top_k: int = 5, use_expansion: bool = True,
                 log_steps: bool = True) -> dict:
        """
        Full retrieval pipeline with logging.
        Returns dict with results + metadata for logging.
        """
        timestamp = datetime.now().isoformat()
        log = {
            "timestamp": timestamp,
            "original_query": query,
            "top_k": top_k,
            "use_expansion": use_expansion,
        }

        # Step 1: Query Expansion
        expanded_query = expand_query(query) if use_expansion else query
        log["expanded_query"] = expanded_query

        # Step 2: Vector search (over-retrieve, then re-rank)
        candidates = self.vs.search(expanded_query, top_k=top_k * 3)
        log["vector_candidates"] = len(candidates)

        # Step 3: Hybrid scoring
        for c in candidates:
            kw_score = self.keyword_score(query, c['text'])
            hybrid = self.alpha * c['similarity_score'] + (1 - self.alpha) * kw_score
            c['keyword_score'] = round(kw_score, 4)
            c['hybrid_score'] = round(hybrid, 4)

        # Step 4: Re-rank by hybrid score
        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        final_results = candidates[:top_k]

        # Step 5: Failure detection
        top_score = final_results[0]['hybrid_score'] if final_results else 0
        retrieval_failed = top_score < 0.05

        log["top_score"] = round(top_score, 4)
        log["retrieval_failed"] = retrieval_failed
        log["results"] = [
            {"chunk_id": r['chunk_id'], "source": r['source'],
             "hybrid_score": r['hybrid_score'], "similarity_score": r['similarity_score'],
             "text_preview": r['text'][:100]}
            for r in final_results
        ]

        if log_steps:
            print(f"\n[Retriever] Query: '{query[:60]}'")
            print(f"[Retriever] Top score: {top_score:.4f} | Failed: {retrieval_failed}")
            for i, r in enumerate(final_results):
                print(f"  #{i+1} [{r['source']}] score={r['hybrid_score']:.4f} | {r['text'][:80]}...")

        if retrieval_failed:
            print(f"[Retriever] ⚠ LOW SCORE DETECTED. FIX: Broadening search with keyword-only fallback.")
            # FIX: fall back to pure keyword search across all chunks
            fallback = sorted(
                self.vs.chunks,
                key=lambda c: self.keyword_score(query, c['text']),
                reverse=True
            )[:top_k]
            for c in fallback:
                c['keyword_score'] = round(self.keyword_score(query, c['text']), 4)
                c['hybrid_score'] = c['keyword_score']
                c['similarity_score'] = 0.0
            log["fallback_used"] = True
            final_results = fallback

        return {"results": final_results, "log": log, "retrieval_failed": retrieval_failed}


if __name__ == "__main__":
    # Quick self-test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_ingestion import ingest_all_data

    chunks = ingest_all_data("../data")
    vs = VectorStore()
    vs.build_index(chunks)
    retriever = HybridRetriever(vs)

    # Test query
    result = retriever.retrieve("Who won the 2020 election in Ashanti Region?", top_k=3)
    print("\nTop result text:")
    print(result['results'][0]['text'][:300])
