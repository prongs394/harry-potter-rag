# Harry Potter RAG Service

A two-stage Retrieval-Augmented Generation (RAG) backend built with FastAPI, ChromaDB, and Groq. The service retrieves relevant context from Harry Potter text documents and makes answers using an LLM.

---

## Architecture Overview

1. **First-Stage Dense Retrieval:**
   - Embeds incoming queries using `sentence-transformers/all-MiniLM-L6-v2`.
   - Performs fast vector similarity search in a persistent `ChromaDB` collection to retrieve candidate chunks.

2. **Context Window Expansion:**
   - Automatically merges adjacent sibling chunks ($i-1$, $i$, $i+1$) to restore paragraph-level narrative context.

3. **Second-Stage Cross-Encoder Reranking:**
   - Reranks expanded candidate passages using `cross-encoder/ms-marco-MiniLM-L-6-v2` to score true semantic alignment with the query.

4. **Generation & Guardrails:**
   - Assembles the top-5 reranked contexts into a strict, context-bound prompt.
   - Generates deterministic completions using Groq API (`llama-3.3-70b-versatile`) with temperature 0.0 to prevent hallucination.

---

## Tech Stack

- **API Framework:** FastAPI, Uvicorn, Pydantic
- **Vector Database:** ChromaDB (Persistent)
- **Embedding Model:** `all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)

---

## Project Structure

```text
.
├── server.py                 # FastAPI application and retrieval pipeline
├── rag_hp.ipynb              # Prototyping, indexing, and vector DB preparation
├── requirements.txt          # Python dependencies
├── .gitignore                # Protects environment keys and vector binaries
└── README.md
