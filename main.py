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

@app.get("/")
def root():
    return {"message": "Welcome to SnapFit API"}
