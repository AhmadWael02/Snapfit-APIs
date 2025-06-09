from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import traceback, json

import models, schemas
from database import get_db
import oauth

router = APIRouter(prefix="/store", tags=["store"])

@router.get("/items")
def get_store_items(db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    try:
        # If user has Store relationship return its clothes json
        storage = db.query(models.Store).filter(models.Store.user_id == current_user.id).first()
        if not storage:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Store not found")
        # storage.clothes is stored as JSON in DB
        return storage
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to fetch store items") 