import os
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# ---------------------------------------------------------
# 1. Configuration & API Key Loading
# ---------------------------------------------------------
# Attempt to load the API key from a local .env file first
API_KEY = None
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")

# Fallback to operating system environment variables
if not API_KEY:
    API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please provide it in .env or environment variables.")

# ---------------------------------------------------------
# 2. FastAPI Application & Request Schema
# ---------------------------------------------------------
app = FastAPI(
    title="Harry Potter RAG Service",
    description="Two-stage retrieval RAG pipeline using ChromaDB, MiniLM Bi-Encoder, Cross-Encoder reranker, and Groq LLM."
)

class QueryRequest(BaseModel):
    query: str

# ---------------------------------------------------------
# 3. Model & Persistent Database Initialization (Run Once)
# ---------------------------------------------------------
print("Loading Bi-Encoder embedding model and Cross-Encoder reranker...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Connect to the local persisted ChromaDB storage
chroma_path = r"D:\projects\harry_potter_rag\chroma_db"
client_db = chromadb.PersistentClient(path=chroma_path)
collection = client_db.get_collection(name="harry_potter_ch1")

# Preload all raw chunks into memory for neighborhood window expansion
raw_data = collection.get()
all_chunks = raw_data["documents"]
total_corpus_chunks = len(all_chunks)

# Initialize Groq client
groq_client = Groq(api_key=API_KEY)
print(f"Service initialized successfully. Indexed chunks available: {total_corpus_chunks}")

# ---------------------------------------------------------
# 4. RAG Query Endpoint
# ---------------------------------------------------------
@app.post("/ask")
def process_query(req: QueryRequest):
    user_query = req.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # Step A: First-Stage Dense Retrieval (Top-100 candidates)
    orig_query_vec = embed_model.encode([user_query]).tolist()
    top_k_results = collection.query(
        query_embeddings=orig_query_vec,
        n_results=100
    )
    retrieved_ids = top_k_results["ids"][0]

    # Step B: Context Window Expansion using Neighbor Chunks (i-1, i, i+1)
    expanded_texts = []
    for cid in retrieved_ids:
        idx = int(cid.split("_")[1])
        start_window = max(0, idx - 1)
        end_window = min(total_corpus_chunks, idx + 2)
        merged_text = " ".join(all_chunks[start_window:end_window])
        expanded_texts.append(merged_text)

    # Step C: Second-Stage Cross-Encoder Reranking
    candidate_pairs = [[user_query, text] for text in expanded_texts]
    rerank_scores = reranker.predict(candidate_pairs)

    scored_candidates = [
        {"id": cid, "score": float(score), "text": exp_text}
        for cid, score, exp_text in zip(retrieved_ids, rerank_scores, expanded_texts)
    ]
    # Sort candidates in descending order of relevance score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Step D: Construct Unified Context from Top-5 Reranked Passages
    top_contexts = [item["text"] for item in scored_candidates[:5]]
    unified_context = "\n\n--- Passage Boundary ---\n\n".join(top_contexts)

    # Step E: Generation Prompt Construction
    rag_prompt = f"""You are a helpful and precise assistant answering questions about the book Harry Potter.
Answer the question based STRICTLY and ONLY on the provided Context below.
If the text does not contain enough information to answer, state clearly: "Information not found in context."

### Context:
{unified_context}

### Question:
{user_query}

### Answer:
"""

    # Step F: Synthesize Response with Groq
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": rag_prompt}],
        model="openai/gpt-oss-120b",
        temperature=0.0,
    )

    final_answer = chat_completion.choices[0].message.content

    return {
        "status": "success",
        "query": user_query,
        "answer": final_answer
    }