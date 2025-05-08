from pydantic import BaseModel, EmailStr
from typing import Optional

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

    class Config:
        orm_mode = True

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
        orm_mode = True


class UserInfo(BaseModel):
    id: int
    email: EmailStr
    user_name: str
    size: Optional[str] = None


class ClassifierResponse(BaseModel):
    subtype: str