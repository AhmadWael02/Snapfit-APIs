from .posts import router as posts_router
from .users import router as users_router
from .auth import router as auth_router
from .classifiers import router as classifiers_router
from .likes import router as likes_router
from .outfits import router as outfits_router
from .brands import router as brands_router

class Config:
    from_attributes = True  # Use this instead of orm_mode=True