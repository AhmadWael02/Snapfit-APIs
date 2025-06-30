from pydantic import BaseModel, EmailStr
from typing import Optional, List
import uvicorn
from fastapi import FastAPI
from datetime import datetime


class PostBase(BaseModel):
    title: str
    body: str
    age: Optional[int] = None


class PostCreate(PostBase):
    pass


class ConsumerCreate(BaseModel):
    email: EmailStr
    password: str
    username: str
    size: Optional[str] = None
    gender: Optional[str] = None


class BrandCreate(BaseModel):
    email: EmailStr
    password: str
    username: str
    description: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: EmailStr
    phone: Optional[str] = None

    class Config:
        from_attributes = True  # Updated from orm_mode=True for Pydantic v2


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class Post(BaseModel):
    title: str
    body: str
    user_id: int
    owner: UserOut

    class Config:
        from_attributes = True  # Updated from orm_mode=True for Pydantic v2


class UserInfo(BaseModel):
    id: int
    email: EmailStr
    user_name: str
    phone: Optional[str] = None
    size: Optional[str] = None

    class Config:
        from_attributes = True


class ClassifierResponse(BaseModel):
    subtype: str


class UserLikeCreate(BaseModel):
    item_id: int


class UserLikeResponse(BaseModel):
    id: int
    user_id: int
    item_id: int

    class Config:
        from_attributes = True  # Updated from orm_mode=True for Pydantic v2


class UserLikesResponse(BaseModel):
    likes: List[UserLikeResponse]

    class Config:
        from_attributes = True  # Updated from orm_mode=True for Pydantic v2


class Clothes(BaseModel):
    id: int
    owner_id: int
    gender: Optional[str] = None
    apparel_type: Optional[str] = None
    subtype: Optional[str] = None
    color: Optional[str] = None
    occasion: Optional[str] = None
    size: Optional[str] = None
    path: Optional[str] = None
    purchase_link: Optional[str] = None
    price: Optional[float] = None
    season: Optional[str] = None
    male_embedding: Optional[str] = None
    female_embedding: Optional[str] = None
    path_3d: Optional[str] = None

    class Config:
        from_attributes = True


class OutfitBase(BaseModel):
    top_id: int
    bottom_id: int
    shoes_id: int
    name: Optional[str] = None
    tags: List[str] = []


class OutfitCreate(OutfitBase):
    pass


class OutfitUpdate(BaseModel):
    name: Optional[str] = None
    tags: Optional[List[str]] = None


class OutfitResponse(OutfitBase):
    id: int
    user_id: int
    is_favorite: bool
    created_at: datetime
    top: Clothes
    bottom: Clothes
    shoes: Clothes

    class Config:
        from_attributes = True


class UserPreferencesBase(BaseModel):
    fit_preference: str = "Regular"
    lifestyle_preferences: List[str] = []
    season_preference: str = "Auto"
    age_group: str = "18-24"
    preferred_colors: List[str] = []
    excluded_categories: List[str] = []


class UserPreferencesUpdate(UserPreferencesBase):
    fit_preference: Optional[str] = None
    lifestyle_preferences: Optional[List[str]] = None
    season_preference: Optional[str] = None
    age_group: Optional[str] = None
    preferred_colors: Optional[List[str]] = None
    excluded_categories: Optional[List[str]] = None


class UserPreferencesResponse(UserPreferencesBase):
    user_id: int

    class Config:
        orm_mode = True


class UserFavoritesResponse(BaseModel):
    id: int
    user_id: int
    item_id: int
    created_at: datetime
    item: dict  # This will contain the clothes item details

    class Config:
        from_attributes = True  # Updated from orm_mode=True for Pydantic v2


class ItemEventCreate(BaseModel):
    item_id: int
    user_id: int
    event_type: str  # 'item_click', 'visit_store', 'recommendation'


class ItemEventStats(BaseModel):
    item_id: int
    item_name: str
    item_photo_url: str
    users_clicked: int
    visit_store: int
    recommended: int


app = FastAPI()


@app.get("/")
def root():
    return {"message": "API is running"}


if __name__ == "__main__":
    uvicorn.run("predictor:app", host="0.0.0.0", port=8000, reload=True)