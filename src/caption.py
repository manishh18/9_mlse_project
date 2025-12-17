# src/caption.py
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch

# choose device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# load once
_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)

def caption_image_local(image_path: str, max_length: int = 64) -> str:
    img = Image.open(image_path).convert("RGB")
    inputs = _processor(images=img, return_tensors="pt").to(DEVICE)
    out = _model.generate(**inputs, max_length=max_length)
    caption = _processor.decode(out[0], skip_special_tokens=True)
    return caption
