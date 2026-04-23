# ACity RAG Chatbot
**CS4241 - Introduction to Artificial Intelligence | 2026**  
**Author:** [Your Full Name]  
**Index Number:** [Your Index Number]  
**Lecturer:** Godwin N. Danso  

---

## 📋 What This Does

A complete **Retrieval-Augmented Generation (RAG)** chatbot built from scratch (no LangChain, no LlamaIndex). It allows you to chat with:
- **Ghana Presidential Election Results** (2008–2020) — CSV dataset
- **Ghana 2025 Budget Statement and Economic Policy** — PDF document

**Every RAG component is manually implemented:**
- TF-IDF embedding pipeline (no paid embedding API)
- Numpy-based vector store (cosine similarity)
- Hybrid search (vector + keyword)
- Query expansion with domain synonyms
- Three prompt templates with hallucination control
- Full pipeline logging at every stage
- Memory-based conversation (Part G)
- User feedback loop (Part G)

---

## 🚀 HOW TO RUN (Step by Step)

### Step 1 — Make sure Python is installed
```bash
python --version
# Should show Python 3.10 or higher
```

### Step 2 — Install dependencies
Open your terminal (or VS Code terminal) in the project folder and run:
```bash
pip install -r requirements.txt
```

If you get a permissions error, try:
```bash
pip install -r requirements.txt --user
```

### Step 3 — Get your FREE Anthropic API key
1. Go to: https://console.anthropic.com
2. Sign up for a free account
3. Click "API Keys" → "Create Key"
4. Copy the key (it starts with `sk-ant-...`)

You DO NOT need to pay — the free tier gives enough credits for this project.

### Step 4 — Add your data files
Make sure these files are in the `data/` folder:
```
data/
  Ghana_Election_Result.csv   ← from the assignment dataset
  budget.pdf                  ← renamed from 2025-Budget-Statement-...pdf
```

### Step 5 — Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at: **http://localhost:8501**

**First run:** The app will build the search index (~60 seconds). After that it loads instantly.

### Step 6 — Use the app
1. Enter your API key in the **sidebar** (left panel)
2. Type a question in the chat box
3. Click **Send →**
4. See the answer, retrieved chunks, and full prompt in the tabs

---

## 📁 Project Structure
```
rag_chatbot/
├── app.py                    # Main Streamlit UI
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── Ghana_Election_Result.csv
│   └── budget.pdf
├── src/
│   ├── data_ingestion.py     # Part A: Data Engineering & Chunking
│   ├── retrieval.py          # Part B: Custom Embedding + Hybrid Search
│   ├── prompt_engineering.py # Part C: Prompt Templates
│   └── rag_pipeline.py       # Part D: Full Pipeline + Logging
├── index/                    # Auto-generated search index
│   ├── embeddings.npy
│   ├── chunks.json
│   └── embedder.pkl
└── logs/                     # Auto-generated pipeline logs
    ├── pipeline_*.jsonl
    ├── prompt_experiments.jsonl
    └── feedback.jsonl
```

---

## 🏗️ Architecture Summary

```
User Query
    │
    ▼
Query Expansion (domain synonyms)
    │
    ▼
Hybrid Retriever
  ├── TF-IDF Vector Search (cosine similarity)
  └── Keyword Search (overlap scoring)
    │
    ▼
Context Window Manager (rank + truncate to 6000 chars)
    │
    ▼
Prompt Builder (v1/v2/v3 templates with hallucination control)
    │
    ▼
Claude API (claude-sonnet-4-20250514)
    │
    ▼
Response + Memory Update + Feedback Logging
```

---

## 📊 Exam Parts Mapping

| Part | File | Implementation |
|------|------|----------------|
| A — Data Engineering | `src/data_ingestion.py` | Cleaning, chunking (400 chars/80 overlap), CSV+PDF |
| B — Custom Retrieval | `src/retrieval.py` | TF-IDF, VectorStore, HybridRetriever, QueryExpansion |
| C — Prompt Engineering | `src/prompt_engineering.py` | 3 templates, context manager, experiment logging |
| D — Full Pipeline | `src/rag_pipeline.py` | End-to-end pipeline with stage logging |
| E — Evaluation | `app.py` Tab 4 | RAG vs LLM comparison, adversarial queries |
| F — Architecture | `app.py` Tab 5 | Diagram + justification |
| G — Innovation | `src/rag_pipeline.py` | Memory-based RAG, Feedback loop, CoT prompting |

---

## 🧪 Sample Questions to Test

**Election questions:**
- Who won the 2020 election in Ashanti Region?
- Which party got the most votes in Greater Accra in 2016?
- What was the NPP vote share in Volta Region in 2020?

**Budget questions:**
- What is Ghana's GDP growth target in the 2025 budget?
- What does the 2025 budget allocate for education?
- What is the fiscal deficit target in the 2025 budget?

**Adversarial (Part E):**
- Who won in the north? *(ambiguous)*
- Did NPP win every single region? *(misleading)*
- What is the population of France? *(out of scope)*

---

## 📧 Submission Details
- **GitHub repo:** `ai_[your_index_number]`
- **Email to:** godwin.danso@acity.edu.gh
- **Subject:** CS4241-Introduction to Artificial Intelligence-2026:[Index] [Name]
- **Add collaborator:** GodwinDansoAcity
