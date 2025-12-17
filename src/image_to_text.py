# src/image_to_text.py
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from src.caption import caption_image_local
from src.clip_embed import embed_image, query_clip_by_image, embed_text
from src.rag_engine import retrieve_chunks, generate_answer_local   # reuse existing functions

# You also have sentence-transformer based retrieval (EMB_MODEL) in rag_engine's retrieve_chunks.
# We'll call both: retrieve_chunks(caption) and clip-based retrieval, then fuse.

def image_to_caption_and_retrieve(image_path: str, k_clip=6, k_caption=6):
    """
    Returns:
      caption: generated caption string
      clip_results: list of {idx, score, meta} for clip retrieval
      caption_retrieved_chunks: list of chunk dicts (from retrieve_chunks using caption)
    """
    # 1) Caption
    caption = caption_image_local(image_path)

    # 2) CLIP retrieval by image
    clip_results = query_clip_by_image(image_path=image_path, k=k_clip)

    # convert clip results meta -> chunk ids; we need full chunk text from chunks.jsonl
    # meta entries have 'id' field (index aligned with chunks)
    clip_chunk_ids = [m["id"] for m in [r["meta"] for r in clip_results]]

    # get chunks for clip ids (use chunks_by_id from rag_engine)
    from src.rag_engine import chunks_by_id
    clip_chunks = [chunks_by_id.get(cid, {}) for cid in clip_chunk_ids]

    # 3) caption-based retrieval (text query into sentence-transformer index)
    caption_retrieved = retrieve_chunks(caption, k=k_caption)

    return {
        "caption": caption,
        "clip_chunks": clip_chunks,
        "caption_chunks": caption_retrieved,
        "clip_scores": clip_results
    }

def fuse_and_rerank(caption_chunks: List[Dict[str,Any]], clip_chunks: List[Dict[str,Any]], clip_scores: List[Dict[str,Any]], top_k=6):
    """
    Fuse results from caption-based retrieval and clip-based retrieval.
    Strategy:
      - Build a candidate set (union of both lists).
      - Score each candidate by: normalized (caption_score_proxy + clip_score_proxy)
      - caption_score_proxy: use semantic similarity between caption and chunk via RAG encoder
      - clip_score_proxy: use clip score if present (already high = better)
      - Finally return top_k chunks (full chunk dicts)
    """
    # Prepare mapping id->chunk
    from src.rag_engine import chunks_by_id
    candidates = {}
    # add caption chunks with proxy score = inverse of distance (sentence-transformer retrieval returns distance)
    for i, c in enumerate(caption_chunks):
        cid = c["id"]
        # convert distance to a similarity proxy (lower distance -> higher proxy)
        proxy = 1.0 / (1.0 + c.get("distance", 1.0))
        candidates[cid] = {"chunk": chunks_by_id.get(cid, {}), "caption_proxy": proxy, "clip_proxy": 0.0}

    # add clip chunks with clip score
    for i, (ck, cs) in enumerate(zip(clip_chunks, clip_scores)):
        cid = ck.get("id")
        if not cid:
            continue
        clip_sim = cs.get("score", 0.0)  # index used IP on normalized vectors, so higher is more similar (0..1+)
        if cid in candidates:
            candidates[cid]["clip_proxy"] = clip_sim
        else:
            candidates[cid] = {"chunk": chunks_by_id.get(cid, {}), "caption_proxy": 0.0, "clip_proxy": clip_sim}

    # compute combined score and sort
    scored = []
    for cid, info in candidates.items():
        # weight both proxies; you can tune weights
        score = 0.6 * info["caption_proxy"] + 0.4 * info["clip_proxy"]
        scored.append((score, info["chunk"]))
    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    top_chunks = [c for s, c in scored_sorted[:top_k]]
    return top_chunks

def image_to_answer(image_path: str, k_clip=6, k_caption=6, top_k=6):
    """
    Full pipeline: caption + clip retrieval + caption retrieval + fusion + LLM answer.
    Returns dict with caption, fused_chunks, and answer string.
    """
    res = image_to_caption_and_retrieve(image_path, k_clip=k_clip, k_caption=k_caption)
    fused = fuse_and_rerank(res["caption_chunks"], res["clip_chunks"], res["clip_scores"], top_k=top_k)
    # LLM expects list of retrieved chunk dicts with text; pass fused chunks
    answer = generate_answer_local(res["caption"], [{"id": c["id"], "title": c["title"], "text": c["text"], "distance": 0.0} for c in fused])
    return {"caption": res["caption"], "fused_chunks": fused, "answer": answer}
