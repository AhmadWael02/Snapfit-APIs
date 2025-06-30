from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from oauth import get_current_user
import models, schemas

router = APIRouter(
    prefix="/outfits",
    tags=["Outfits"]
)

@router.get("/", response_model=List[schemas.OutfitResponse])
def get_user_outfits(db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    outfits = db.query(models.Outfit).filter(models.Outfit.user_id == current_user.id).all()
    print(f"DEBUG: get_user_outfits for user_id={current_user.id}, found {len(outfits)} outfits: {[o.id for o in outfits]}")
    return outfits

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=schemas.OutfitResponse)
def create_outfit(outfit: schemas.OutfitCreate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    # Check if all clothes items exist and belong to the user
    top = db.query(models.Clothes).filter(models.Clothes.id == outfit.top_id).first()
    bottom = db.query(models.Clothes).filter(models.Clothes.id == outfit.bottom_id).first()
    shoes = db.query(models.Clothes).filter(models.Clothes.id == outfit.shoes_id).first()
    
    if not top or not bottom or not shoes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                           detail="One or more clothing items not found")
    
    new_outfit = models.Outfit(
        user_id=current_user.id,
        top_id=outfit.top_id,
        bottom_id=outfit.bottom_id,
        shoes_id=outfit.shoes_id,
        name=outfit.name,
        tags=outfit.tags
    )
    
    db.add(new_outfit)
    db.commit()
    db.refresh(new_outfit)
    
    return new_outfit

@router.get("/{outfit_id}", response_model=schemas.OutfitResponse)
def get_outfit(outfit_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    outfit = db.query(models.Outfit).filter(
        models.Outfit.id == outfit_id,
        models.Outfit.user_id == current_user.id
    ).first()
    
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                           detail=f"Outfit with id {outfit_id} not found")
    
    return outfit

@router.post("/{outfit_id}/toggle-favorite", response_model=schemas.OutfitResponse)
def toggle_outfit_favorite(outfit_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    outfit = db.query(models.Outfit).filter(
        models.Outfit.id == outfit_id,
        models.Outfit.user_id == current_user.id
    ).first()
    
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                           detail=f"Outfit with id {outfit_id} not found")
    
    outfit.is_favorite = not outfit.is_favorite
    db.commit()
    db.refresh(outfit)
    
    return outfit

@router.put("/{outfit_id}", response_model=schemas.OutfitResponse)
def update_outfit(outfit_id: int, outfit_update: schemas.OutfitUpdate, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    outfit = db.query(models.Outfit).filter(
        models.Outfit.id == outfit_id,
        models.Outfit.user_id == current_user.id
    ).first()
    
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                           detail=f"Outfit with id {outfit_id} not found")
    
    # Update fields if provided
    if outfit_update.name is not None:
        outfit.name = outfit_update.name
    if outfit_update.tags is not None:
        outfit.tags = outfit_update.tags
    
    db.commit()
    db.refresh(outfit)
    
    return outfit

@router.delete("/{outfit_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_outfit(outfit_id: int, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    outfit_query = db.query(models.Outfit).filter(
        models.Outfit.id == outfit_id,
        models.Outfit.user_id == current_user.id
    )
    
    outfit = outfit_query.first()
    
    if not outfit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                           detail=f"Outfit with id {outfit_id} not found")
    
    outfit_query.delete(synchronize_session=False)
    db.commit()
    
    return None 