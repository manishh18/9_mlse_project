# src/clip_embed.py
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import faiss
import torch
from PIL import Image

ROOT = Path.cwd()
EMB_DIR = ROOT / "embeddings"

# model name must match the one used in notebook
CLIP_TEXT_MODEL = "clip-ViT-B-32"
_clip_model = SentenceTransformer(CLIP_TEXT_MODEL)  # supports both text & image encode if using proper model

# NOTE: sentence-transformers CLIP models often have both image and text encoders available via encode()
# But depending on model, you may need to pass images as numpy arrays / PIL Images.
# We'll implement embed_text and embed_image wrappers.

def embed_text(texts, normalize=True):
    """
    texts: list[str] or single str
    returns: numpy array (n, d) float32, normalized if normalize=True
    """
    if isinstance(texts, str):
        texts = [texts]
    embs = _clip_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
    return embs.astype("float32")

def embed_image(image_path, normalize=True):
    """
    Get CLIP image embedding for a single image path.
    """
    # sentence-transformers can accept PIL images or numpy arrays
    img = Image.open(image_path).convert("RGB")
    embs = _clip_model.encode([img], convert_to_numpy=True, show_progress_bar=False)
    if normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
    return embs.astype("float32")[0]

# Load CLIP FAISS index & meta
def load_clip_index():
    index_file = EMB_DIR / "clip_faiss_index.bin"
    meta_file = EMB_DIR / "clip_text_chunks_meta.json"
    if not index_file.exists():
        raise FileNotFoundError(f"CLIP index not found at {index_file}. Run 04-clip-index.ipynb first.")
    index = faiss.read_index(str(index_file))
    meta = []
    import json
    with open(meta_file, encoding="utf8") as f:
        meta = json.load(f)
    return index, meta

# Query helper: takes image path or image embedding and returns top-k chunks (meta indices)
def query_clip_by_image(image_path=None, image_emb=None, k=6):
    if image_emb is None:
        image_emb = embed_image(image_path)
    index, meta = load_clip_index()
    # index expects shape (1, d)
    D, I = index.search(np.array([image_emb], dtype="float32"), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta_rec = meta[idx]
        results.append({"idx": int(idx), "score": float(score), "meta": meta_rec})
    return results
