from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional
# import psycopg2
# from psycopg2.extras import RealDictCursor
import models
from database import engine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
# Import routers directly
from routers.posts import router as posts_router
from routers.users import router as users_router
from routers.auth import router as auth_router
from routers.classifiers import router as classifiers_router
from routers.likes import router as likes_router
# Newly added routers
from routers.clothes import router as clothes_router
from routers.shop import router as shop_router
from routers.store import router as store_router
from routers.brands import router as brands_router
from routers.ai import router as ai_router
from routers.outfits import router as outfits_router
from routers.user import router as user_router
from config import settings
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Create directories for static files if they don't exist
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/images/clothes", exist_ok=True)
os.makedirs("static/images/profile_pics", exist_ok=True)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace with your Flutter app's origin)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include routers directly
app.include_router(posts_router)
app.include_router(users_router)
app.include_router(auth_router)
app.include_router(classifiers_router)
app.include_router(likes_router)
app.include_router(clothes_router)
app.include_router(shop_router)
app.include_router(store_router)
app.include_router(brands_router)
app.include_router(ai_router)
app.include_router(outfits_router)
app.include_router(user_router)

# Load data with adjusted paths
with open('../snapfit_v1/assets/models/fashion_embeddings/documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)
embeddings = np.load('../snapfit_v1/assets/models/fashion_embeddings/embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')

HF_API_KEY = settings.hf_api_key
HF_MODEL = 'HuggingFaceH4/zephyr-7b-beta'

def hf_chat(prompt, model=HF_MODEL):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100, "temperature": 0.7}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        elif isinstance(result, list) and 'generated_text' in result[0].get('generated_text', {}):
            return result[0]['generated_text']
        elif isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        else:
            return str(result)
    else:
        return f"Sorry, I couldn't generate a response. (HF API error: {response.text})"

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.query
    query_emb = model.encode([user_query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:3]  # Top 3
    context = '\n'.join([documents[i]['page_content'] for i in top_idx])
    prompt = (
        "You are a helpful AI fashion stylist. "
        "Given the context below, answer the user's question as briefly and directly as possible. "
        "Do NOT repeat the context, instructions, or question. Only output the answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )
    answer = hf_chat(prompt)
    # Remove everything before and including 'Answer:'
    if "Answer:" in answer:
        answer = answer.split("Answer:", 1)[1].strip()
    # Remove any repeated context, prompt lines, or markdown headers
    lines = [line.strip() for line in answer.splitlines() if line.strip() and not line.strip().lower().startswith("context:") and not line.strip().startswith("#")]
    answer = lines[0] if lines else ""
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Welcome to SnapFit API"}
