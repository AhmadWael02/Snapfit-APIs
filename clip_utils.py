from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import torch
import os

# Load the CLIP model once
clip_model = SentenceTransformer("clip-ViT-B-32")

def get_clip_embedding(image_path: str) -> str:
    """
    Given an image path, load the image and return the CLIP embedding as a JSON string (for DB storage).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    # SentenceTransformer expects a list of images
    embedding = clip_model.encode([img])[0]
    # Convert to list for JSON serialization
    return np.array(embedding, dtype=np.float32).tolist() 