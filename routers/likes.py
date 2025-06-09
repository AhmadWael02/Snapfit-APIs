from fastapi import HTTPException, status, Depends, APIRouter
from sqlalchemy.orm import Session
import sqlalchemy
import traceback

# Fix import issues
import sys
import os
# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import the local modules
from database import get_db
import models
import oauth
import schemas

router = APIRouter(prefix="/likes", tags=["likes"])

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=schemas.UserLikeResponse)
def create_like(like: schemas.UserLikeCreate, db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    try:
        # Check if item exists
        item = db.query(models.Clothes).filter(models.Clothes.id == like.item_id).first()
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Item with id: {like.item_id} not found")
        
        # Check if user already liked this item
        existing_like = db.query(models.UserLikes).filter(
            models.UserLikes.user_id == current_user.id,
            models.UserLikes.item_id == like.item_id
        ).first()
        
        if existing_like:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"User {current_user.id} already liked item {like.item_id}")
        
        # Create new like
        new_like = models.UserLikes(user_id=current_user.id, item_id=like.item_id)
        db.add(new_like)
        db.commit()
        db.refresh(new_like)
        
        return new_like
    
    except HTTPException as e:
        db.rollback()
        raise e
    
    except Exception as e:
        db.rollback()
        print(f"An unexpected error occurred in create_like: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while creating the like."
        )

@router.get("/", response_model=list[schemas.UserLikeResponse])
def get_user_likes(db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    try:
        likes = db.query(models.UserLikes).filter(models.UserLikes.user_id == current_user.id).all()
        return likes
    
    except Exception as e:
        print(f"An unexpected error occurred in get_user_likes: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while retrieving the likes."
        )

@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_like(item_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    try:
        like_query = db.query(models.UserLikes).filter(
            models.UserLikes.user_id == current_user.id,
            models.UserLikes.item_id == item_id
        )
        
        like = like_query.first()
        
        if not like:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Like not found for user {current_user.id} and item {item_id}"
            )
        
        like_query.delete(synchronize_session=False)
        db.commit()
        
        return
    
    except HTTPException as e:
        db.rollback()
        raise e
    
    except Exception as e:
        db.rollback()
        print(f"An unexpected error occurred in delete_like: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while deleting the like."
        ) 