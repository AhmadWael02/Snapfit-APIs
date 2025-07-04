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

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

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

# Chat memory buffer (windowed)
chat_memory = {}  # user_id -> list of {role: 'user'/'assistant', 'content': str}
MEMORY_WINDOW_SIZE = 6  # Number of (user, assistant) pairs to keep
MAX_TOKENS = 120  # Maximum tokens/words in the reply

HF_API_KEY = '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'
print("HF_API_KEY loaded:", HF_API_KEY)  # DEBUG PRINT
HF_MODEL = 'gpt2'

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # Use the larger, better model by default

SERPAPI_KEY = settings.serpapi_key

EMBEDDING_SIMILARITY_THRESHOLD = 0.4  # Lowered threshold to allow short/greeting matches

def ollama_chat(prompt, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    print("[OLLAMA DEBUG] URL:", OLLAMA_URL)
    print("[OLLAMA DEBUG] Payload:", payload)
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        print("[OLLAMA DEBUG] Response status:", response.status_code)
        print("[OLLAMA DEBUG] Response text:", response.text)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "")
        else:
            return None
    except Exception as e:
        print("[OLLAMA DEBUG] Exception:", e)
        return None

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

def web_search_serpapi(query):
    print("[WEB SEARCH DEBUG] Query:", query)
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": 3
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        print("[WEB SEARCH DEBUG] Response status:", response.status_code)
        if response.status_code == 200:
            data = response.json()
            # Try to get the best answer from answer_box or organic results
            if "answer_box" in data and "answer" in data["answer_box"]:
                return data["answer_box"]["answer"]
            if "organic_results" in data and data["organic_results"]:
                return data["organic_results"][0].get("snippet", "No snippet found.")
            return "Sorry, I couldn't find an answer on the web."
        else:
            return "Sorry, web search failed."
    except Exception as e:
        print("[WEB SEARCH DEBUG] Exception:", e)
        return "Sorry, web search failed."

def build_llama3_prompt(system_message, history, user_message):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system_message + "<|eot_id|>"
    for msg in history:
        role = msg['role']
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{msg['content']}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

FASHION_KEYWORDS = [
    'fashion', 'style', 'outfit', 'clothes', 'clothing', 'wear', 'dress', 'skirt', 'shirt', 'pants', 'jeans', 'jacket', 'coat', 'suit', 'shoes', 'sneakers', 'boots', 'heels', 'accessory', 'accessories', 'bag', 'purse', 'scarf', 'hat', 'trend', 'season', 'color', 'colors', 'match', 'matching', 'wardrobe', 'look', 'designer', 'brand', 'runway', 'fit', 'fabric', 'material', 'pattern', 'print', 'casual', 'formal', 'event', 'wedding', 'party', 'work', 'workout', 'weekend', 'everyday', 'occasion', 'shopping', 'shop', 'closet', 'stylist', 'recommend', 'suggest', 'coordinate', 'ensemble', 'combination', 'mix', 'blend', 'fashionable', 'unfashionable', 'in style', 'out of style', 'dress code', 'attire', 'garment', 'apparel', 'suit', 'tie', 'blouse', 't-shirt', 'sweater', 'hoodie', 'shorts', 'swimsuit', 'swimwear', 'lingerie', 'outerwear', 'denim', 'leather', 'silk', 'cotton', 'linen', 'wool', 'polyester', 'nylon', 'velvet', 'velour', 'satin', 'chiffon', 'lace', 'sequin', 'embroidery', 'tailor', 'tailored', 'fit', 'fitting', 'oversized', 'slim', 'skinny', 'baggy', 'loose', 'tight', 'crop', 'cropped', 'long', 'short', 'mini', 'maxi', 'midi', 'vintage', 'retro', 'modern', 'classic', 'minimal', 'boho', 'bohemian', 'preppy', 'edgy', 'streetwear', 'athleisure', 'couture', 'haute', 'couture', 'ready-to-wear', 'bespoke', 'custom', 'runway', 'catwalk', 'collection', 'season', 'spring', 'summer', 'fall', 'autumn', 'winter', 'lookbook', 'inspiration', 'inspire', 'influencer', 'model', 'fashionista', 'icon', 'statement', 'piece', 'wardrobe', 'closet', 'outfitting', 'styling', 'personal shopper', 'personal stylist', 'fashion advice', 'fashion tip', 'fashion tips', 'fashion advice', 'fashion help', 'fashion question', 'fashion answer', 'fashion suggestion', 'fashion recommendation', 'fashion recommend', 'fashion suggest', 'fashion coordinate', 'fashion ensemble', 'fashion combination', 'fashion mix', 'fashion blend']

def is_fashion_question(text):
    text_lower = text.lower()
    return any(word in text_lower for word in FASHION_KEYWORDS)

def truncate_text(text, max_tokens=MAX_TOKENS):
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + '...'
    return text

@app.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.query
    user_id = request.user_id or "default"
    if user_id not in chat_memory:
        chat_memory[user_id] = []
    chat_memory[user_id].append({"role": "user", "content": user_query})
    history = chat_memory[user_id][-MEMORY_WINDOW_SIZE:]
    system_message = "You are a helpful AI fashion stylist."
    # Try both fashion and chat embeddings for context retrieval first
    query_emb = model.encode([user_query])
    fashion_sims = cosine_similarity(query_emb, fashion_embeddings)[0]
    chat_sims = cosine_similarity(query_emb, chat_embeddings)[0]
    fashion_top_idx = np.argsort(fashion_sims)[::-1][:3]
    chat_top_idx = np.argsort(chat_sims)[::-1][:3]
    top_fashion_score = fashion_sims[fashion_top_idx[0]]
    top_chat_score = chat_sims[chat_top_idx[0]]
    fashion_context = '\n'.join([fashion_documents[i]['page_content'] for i in fashion_top_idx])
    chat_context = '\n'.join([chat_documents[i]['page_content'] for i in chat_top_idx])
    print(f"[CHAT ENDPOINT DEBUG] Top fashion embedding similarity: {top_fashion_score}")
    print(f"[CHAT ENDPOINT DEBUG] Top chat embedding similarity: {top_chat_score}")
    answer = None
    # Use the context with the highest similarity if above threshold, and pass it to LLM for generation
    use_context = None
    if top_fashion_score > top_chat_score and top_fashion_score > EMBEDDING_SIMILARITY_THRESHOLD:
        use_context = fashion_context
    elif top_chat_score >= top_fashion_score and top_chat_score > EMBEDDING_SIMILARITY_THRESHOLD:
        use_context = chat_context
    if use_context:
        # Build a RAG prompt for Llama 3
        rag_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful AI fashion stylist. Use the following context to answer the user's question.\n"
            f"Context:\n{use_context}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_query}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        print("[CHAT ENDPOINT DEBUG] Using RAG prompt with context for Ollama.")
        answer = ollama_chat(rag_prompt)
    else:
        print("[CHAT ENDPOINT DEBUG] No relevant context found (similarity too low), trying Ollama with normal prompt.")
        prompt = build_llama3_prompt(system_message, history[:-1], user_query)
        answer = ollama_chat(prompt)
    if not answer or len(answer.strip()) < 3:
        print("[CHAT ENDPOINT DEBUG] Ollama failed, falling back to SerpAPI web search.")
        answer = web_search_serpapi(user_query)
    if not answer or "Sorry" in answer or len(answer.strip()) < 3:
        print("[CHAT ENDPOINT DEBUG] SerpAPI failed, falling back to Hugging Face API.")
        prompt = build_llama3_prompt(system_message, history[:-1], user_query)
        answer = hf_chat(prompt)
    answer = truncate_text(answer)
    chat_memory[user_id].append({"role": "assistant", "content": answer})
    if len(chat_memory[user_id]) > MEMORY_WINDOW_SIZE:
        chat_memory[user_id] = chat_memory[user_id][-MEMORY_WINDOW_SIZE:]
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Welcome to SnapFit API"}
