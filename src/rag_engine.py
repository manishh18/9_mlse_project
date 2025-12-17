# rag_engine.py
# RAG engine: load FAISS index + embeddings + chunks, provide retrieve_chunks()
# and generate_answer_local() using a local seq2seq LLM (default: google/flan-t5-small).
# Place this file in src/ and import in your Flask app: from src.rag_engine import retrieve_chunks, generate_answer_local

import os
import json
from pathlib import Path
from typing import List, Dict, Any

# --- Paths (adjust if your project layout is different) ---
ROOT = Path.cwd()
EMB_DIR = ROOT / "embeddings"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

# --- Load chunks (full text) ---
chunks = []
if CHUNKS_FILE.exists():
    with open(CHUNKS_FILE, encoding="utf8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
else:
    raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

chunks_by_id = {c["id"]: c for c in chunks}

# --- Load metadata (aligned with embeddings/index) ---
meta = []
meta_file = EMB_DIR / "metadata.jsonl"
if meta_file.exists():
    with open(meta_file, encoding="utf8") as f:
        meta = [json.loads(l) for l in f.read().splitlines() if l.strip()]
else:
    raise FileNotFoundError(f"Metadata file not found: {meta_file}")

# --- Load FAISS index and optionally embeddings (for reranking) ---
import numpy as np
import faiss

index_file = EMB_DIR / "faiss_index.bin"
emb_np_file = EMB_DIR / "embeddings.npy"
if not index_file.exists():
    raise FileNotFoundError(f"FAISS index not found: {index_file}")

index = faiss.read_index(str(index_file))
embeddings = None
if emb_np_file.exists():
    embeddings = np.load(str(emb_np_file))

# --- Embedding encoder (same model used during index creation) ---
from sentence_transformers import SentenceTransformer
EMB_MODEL = "all-MiniLM-L6-v2"
_encoder = SentenceTransformer(EMB_MODEL)

# --- Retrieval function ---
def retrieve_chunks(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Return top-k chunks for the query. Each result includes id, title, distance, text."""
    q_emb = _encoder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype('float32'), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = meta[idx]
        c = chunks_by_id.get(m['id'], {})
        results.append({
            "id": m.get('id'),
            "title": m.get('title'),
            "distance": float(dist),
            "text": c.get('text','')
        })
    return results

# --- Local LLM for generation (default: seq2seq FLAN-T5 small) ---
# The code tries to load with device_map='auto' (requires accelerate). If unavailable,
# it falls back to CPU. Adjust MODEL_NAME to use a different model.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = os.environ.get("RAG_LLM_MODEL", "google/flan-t5-small")

print(f"Loading LLM model: {MODEL_NAME} (this may take a moment on first run)")

def _load_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map="auto", low_cpu_mem_usage=True)
        device = next(model.parameters()).device
        print("Loaded model with device_map; device:", device)
    except Exception as e:
        print("device_map auto failed or accelerate not installed; falling back to CPU. Error:", e)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        model = model.to("cpu")
        device = torch.device("cpu")
        print("Loaded model on CPU (slow).")
    return tokenizer, model

TOKENIZER, MODEL = _load_model(MODEL_NAME)
MODEL_DEVICE = next(MODEL.parameters()).device

# --- Utility: build prompt with retrieved context and trim if necessary ---

def _build_prompt(question: str, retrieved: List[Dict[str, Any]], max_context_chars: int = 6000) -> str:
    """Construct prompt for seq2seq model using retrieved chunks. Trims context if too long."""
    context_pieces = []
    for i, r in enumerate(retrieved):
        piece = f"Chunk {i+1} ({r.get('title')}):\n{r.get('text')}"
        context_pieces.append(piece)
    context = "\n\n".join(context_pieces)
    # If context too large, truncate from the end
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    prompt = (
        "You are an expert on Indian historical monuments."
        " Use ONLY the information in the Context section to answer the question."
        " If the information is not present, say 'I don't know based on the provided data.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt

# --- Generation function using local model ---

def generate_answer_local(question: str, retrieved: List[Dict[str, Any]], max_new_tokens: int = 150) -> str:
    """Generate an answer using the local seq2seq model and retrieved chunks."""
    prompt = _build_prompt(question, retrieved)
    inputs = TOKENIZER(prompt, return_tensors="pt")
    # move to model device
    inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    # attempt to extract text after 'Answer:' if model repeats prompt
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()

# --- Convenience pipeline function ---

def rag_query(question: str, k: int = 4) -> Dict[str, Any]:
    retrieved = retrieve_chunks(question, k=k)
    answer = generate_answer_local(question, retrieved)
    return {"question": question, "answer": answer, "retrieved": retrieved}

# If this file is run directly, do a small smoke test (only if embeddings exist)
if __name__ == "__main__":
    q = "When was the Taj Mahal built and by whom?"
    print("Running smoke test: retrieve + generate for sample question")
    res = rag_query(q, k=3)
    print("Answer:\n", res['answer'])
    print("Retrieved chunks:")
    for r in res['retrieved']:
        print("-", r['id'], r['title'], "(dist=", r['distance'], ")")
