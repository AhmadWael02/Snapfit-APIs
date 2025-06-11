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
from datetime import datetime
from typing import Optional

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
    fashion_documents = json.load(f)
fashion_embeddings = np.load('../snapfit_v1/assets/models/fashion_embeddings/embeddings.npy')

with open('../snapfit_v1/assets/models/chat_embeddings/documents.json', 'r', encoding='utf-8') as f:
    chat_documents = json.load(f)
chat_embeddings = np.load('../snapfit_v1/assets/models/chat_embeddings/embeddings.npy')

model = SentenceTransformer('all-MiniLM-L6-v2')

# Chat memory buffer
chat_memory = {}

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
    user_id: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.query
    user_id = request.user_id or "default"
    
    # Initialize or get user's chat memory
    if user_id not in chat_memory:
        chat_memory[user_id] = []
    
    # Get relevant fashion context
    query_emb = model.encode([user_query])
    fashion_sims = cosine_similarity(query_emb, fashion_embeddings)[0]
    fashion_top_idx = np.argsort(fashion_sims)[::-1][:3]
    fashion_context = '\n'.join([fashion_documents[i]['page_content'] for i in fashion_top_idx])
    
    # Get relevant chat context
    chat_sims = cosine_similarity(query_emb, chat_embeddings)[0]
    chat_top_idx = np.argsort(chat_sims)[::-1][:2]
    chat_context = '\n'.join([chat_documents[i]['page_content'] for i in chat_top_idx])
    
    # Combine contexts and chat history
    chat_history = '\n'.join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_memory[user_id][-3:]])
    
    prompt = (
        "You are a helpful AI fashion stylist. "
        "Given the following contexts and chat history, answer the user's question as briefly and directly as possible. "
        "Do NOT repeat the context, instructions, or question. Only output the answer.\n\n"
        f"Fashion Context:\n{fashion_context}\n\n"
        f"Chat Context:\n{chat_context}\n\n"
        f"Recent Chat History:\n{chat_history}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )
    
    answer = hf_chat(prompt)
    
    # Clean up the answer
    if "Answer:" in answer:
        answer = answer.split("Answer:", 1)[1].strip()
    lines = [line.strip() for line in answer.splitlines() if line.strip() and not line.strip().lower().startswith("context:") and not line.strip().startswith("#")]
    answer = lines[0] if lines else ""
    
    # Update chat memory
    chat_memory[user_id].append({
        "user": user_query,
        "assistant": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 10 messages in memory
    if len(chat_memory[user_id]) > 10:
        chat_memory[user_id] = chat_memory[user_id][-10:]
    
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Welcome to SnapFit API"}
