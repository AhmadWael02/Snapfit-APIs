from fastapi import APIRouter, Depends, status, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import get_db
import models
import utils
import oauth

router = APIRouter(tags=["Authentications"])

@router.post("/login", status_code=status.HTTP_200_OK)
def login(user_creds: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    user = db.query(models.User).filter(models.User.email == user_creds.username).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid credentials")
    
    password_entry = db.query(models.User.password).filter(models.User.id == user.id).first()

    if not utils.verify(user_creds.password, password_entry.password):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid credentials")
    
    access_token = oauth.create_access_token(data={"user_email": user.email})

    return {"access_token": access_token, "token_type": "bearer"}