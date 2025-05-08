from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import models
from database import engine
from routers import posts, users, auth, classifiers
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace with your Flutter app's origin)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.include_router(posts.router)
app.include_router(users.router)
app.include_router(auth.router)
app.include_router(classifiers.router)
