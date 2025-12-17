# src/t2i.py
"""
Text -> Image utilities for the RAG multimodal project (FREE local-only version).

This simplified module **only** supports local generation via Hugging Face Diffusers
and does **not** include any hosted/paid API fallbacks (HuggingFace Inference or Replicate).

Usage:
    from src.t2i import generate_image, generate_image_with_facts
    out = generate_image("taj mahal at sunrise, watercolor", out_name="out.png")

Notes:
- Local generation is free but requires downloading model weights (first run) and
  having enough disk space (~4GB+) and RAM. A GPU (CUDA) or Apple MPS will speed up
  generation; CPU-only is slow but works for small tests.
- Install required libraries before use: diffusers, transformers, accelerate, safetensors, and torch.

"""

from pathlib import Path
from typing import List, Dict, Any, Optional

OUT_DIR = Path("data/generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Local diffusers path (ONLY local, free) --------
_local_pipe = None
_local_model = "runwayml/stable-diffusion-v1-5"  # change if you prefer another local model

def _load_local_pipe(model_name: str = None):
    """Load and return a StableDiffusionPipeline. Caches in module global.
    Raises helpful error messages if required libraries are missing.
    """
    global _local_pipe
    if _local_pipe is not None:
        return _local_pipe
    try:
        from diffusers import StableDiffusionPipeline
        import torch
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for local generation. Install with: "
            "pip install diffusers transformers accelerate safetensors" 

            "and install torch appropriate for your platform. See https://pytorch.org for details.") from e

    model_name = model_name or _local_model

    # choose device
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"

    # load pipeline
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to download/load model {model_name}. Ensure you have network access and enough disk space. Error: {e}") from e

    try:
        pipe = pipe.to(device)
    except Exception:
        pipe = pipe.to("cpu")

    _local_pipe = pipe
    return _local_pipe

# -------- Public API functions (local only) --------

def generate_image(prompt: str, out_name: Optional[str] = None, steps: int = 30, seed: Optional[int] = None) -> str:
    """
    Generate an image from `prompt` locally using diffusers and save to data/generated/.
    Returns the saved filepath.
    """
    if out_name:
        out_name = str(out_name)
    else:
        safe = abs(hash(prompt)) % (10**8)
        out_name = f"gen_{safe}.png"
    out_path = OUT_DIR / out_name

    pipe = _load_local_pipe()
    try:
        import torch
        generator = None
        # use the pipe's device for generator if available
        device = pipe.device if hasattr(pipe, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            try:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                generator = gen
            except Exception:
                generator = None
        result = pipe(prompt, num_inference_steps=steps, generator=generator)
        img = result.images[0]
        img.save(out_path)
        return str(out_path)
    except Exception as e:
        raise RuntimeError("Local generation failed: " + str(e)) from e


def generate_image_with_facts(prompt: str, facts: List[Dict[str, Any]], out_name: Optional[str] = None, steps: int = 30) -> str:
    """
    Enrich the user's prompt with a short set of factual phrases extracted from retrieved chunks,
    then generate an image from the enriched prompt using the local pipeline.

    facts: list of chunk dicts with fields 'title' and 'text'
    """
    # create short factual prefix
    fact_phrases = []
    for f in facts[:3]:
        text = f.get('text','')
        sentence = text.split('.')[:1][0]
        words = ' '.join(sentence.split()[:20])
        if words:
            fact_phrases.append(words.strip())
    if fact_phrases:
        facts_str = ' | '.join(fact_phrases)
        enriched = f"{prompt} --context: {facts_str}"
    else:
        enriched = prompt
    return generate_image(enriched, out_name=out_name, steps=steps)

# -------- Simple CLI test when run directly --------
if __name__ == '__main__':
    print('Local-only T2I module')
    p = "Taj Mahal at sunrise, watercolor painting, soft light, high detail"
    out = generate_image(p)
    print('Saved image to', out)
