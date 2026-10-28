# Harry Potter RAG Service

A two-stage Retrieval-Augmented Generation (RAG) backend built with FastAPI, ChromaDB, and Groq. The service retrieves relevant context from Harry Potter text documents and generates answers using an LLM.

---

## How It Works

1. Step 1: Dense Retrieval
   - Encodes user queries using sentence-transformers/all-MiniLM-L6-v2.
   - Performs vector similarity search against candidate chunks stored in a local ChromaDB collection.

2. Step 2: Context Window Expansion
   - Merges adjacent sibling chunks (i-1, i, i+1) to restore paragraph-level context.

3. Step 3: Cross-Encoder Reranking
   - Reranks expanded candidate passages using cross-encoder/ms-marco-MiniLM-L-6-v2 to score true semantic alignment.

4. Step 4: Answer Generation
   - Assembles the top-5 reranked contexts into a strict prompt.
   - Generates deterministic completions via Groq API (llama-3.3-70b-versatile) with temperature 0.0.

---

## Tech Stack

- API Framework: FastAPI, Uvicorn, Pydantic
- Vector Database: ChromaDB (Persistent)
- Embedding Model: all-MiniLM-L6-v2
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM Engine: Groq API (llama-3.3-70b-versatile)

---

## Project Structure

- server.py: FastAPI application and retrieval pipeline
- rag_hp.ipynb: Prototyping, indexing, and vector DB preparation
- project_requirements.txt: Python dependencies
- .gitignore: Protects environment keys and vector binaries
- README.md: Documentation

---

## Setup & Running Locally

1. Clone the repository:
   git clone https://github.com/prongs394/harry-potter-rag.git
   cd harry-potter-rag

2. Set up environment variables:
   Get a free API key from https://console.groq.com/keys
   Create a .env file in the root directory and add:
   GROQ_API_KEY=your_actual_groq_api_key_here

3. Install dependencies:
   pip install -r project_requirements.txt

4. Prepare Data & Build Vector Database:
   - Provide your own Harry Potter book files (PDF / TXT). (No, the copyrighted books aren't in this repo—Accio Book won't fetch them for you either).
   - Set the book file path in `rag_hp.ipynb` and run it once to chunk the text, generate embeddings, and build the local `chroma_db/` directory.

5. Start the service:
   python -m uvicorn server:app --host 0.0.0.0 --port 8000

---

## API Usage

- Interactive API Docs (Swagger UI):
  Visit http://localhost:8000/docs in your browser.

- Sample cURL Request:
  curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"query\": \"Who are Harry's best friends?\"}"

- Sample Response:
  {"status": "success", "query": "Who are Harry's best friends?", "answer": "Harry's best friends are Ron Weasley and Hermione Granger."}
