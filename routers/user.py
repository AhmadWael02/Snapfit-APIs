from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from database import get_db
from models import User, UserPreferences, UserFavorites, Clothes
from schemas import UserPreferencesUpdate, UserFavoritesResponse, UserPreferencesResponse
from oauth import get_current_user

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    preferences = db.query(UserPreferences).filter(UserPreferences.user_id == current_user.id).first()
    if not preferences:
        # Create default preferences if none exist
        preferences = UserPreferences(
            user_id=current_user.id,
            fit_preference="Regular",
            lifestyle_preferences=[],
            season_preference="Auto",
            age_group="18-24",
            preferred_colors=[],
            excluded_categories=[]
        )
        db.add(preferences)
        db.commit()
        db.refresh(preferences)
    return preferences

@router.put("/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    preferences: UserPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_preferences = db.query(UserPreferences).filter(UserPreferences.user_id == current_user.id).first()
    if not db_preferences:
        db_preferences = UserPreferences(user_id=current_user.id)
        db.add(db_preferences)
    
    for key, value in preferences.dict(exclude_unset=True).items():
        setattr(db_preferences, key, value)
    
    db.commit()
    db.refresh(db_preferences)
    return db_preferences

@router.get("/favorites", response_model=List[UserFavoritesResponse])
async def get_user_favorites(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    favorites = db.query(UserFavorites).options(joinedload(UserFavorites.item)).filter(UserFavorites.user_id == current_user.id).all()
    result = []
    for favorite in favorites:
        item = favorite.item
        item_dict = None
        if item:
            item_dict = {
                "id": item.id,
                "owner_id": item.owner_id,
                "gender": item.gender,
                "apparel_type": item.apparel_type,
                "subtype": item.subtype,
                "color": item.color,
                "occasion": item.occasion,
                "size": item.size,
                "path": item.path,
                "purchase_link": item.purchase_link,
                "price": item.price,
            }
        favorite_dict = {
            "id": favorite.id,
            "user_id": favorite.user_id,
            "item_id": favorite.item_id,
            "created_at": favorite.created_at,
            "item": item_dict,
        }
        result.append(favorite_dict)
    return result

@router.post("/favorites/{item_id}")
async def toggle_favorite(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if item exists
    item = db.query(Clothes).filter(Clothes.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check if already favorited
    existing_favorite = db.query(UserFavorites).filter(
        UserFavorites.user_id == current_user.id,
        UserFavorites.item_id == item_id
    ).first()
    
    if existing_favorite:
        # Remove favorite
        db.delete(existing_favorite)
        db.commit()
        return {"message": "Item removed from favorites"}
    else:
        # Add favorite
        new_favorite = UserFavorites(
            user_id=current_user.id,
            item_id=item_id
        )
        db.add(new_favorite)
        db.commit()
        return {"message": "Item added to favorites"}

@router.delete("/favorites/{item_id}")
async def remove_favorite(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    favorite = db.query(UserFavorites).filter(
        UserFavorites.user_id == current_user.id,
        UserFavorites.item_id == item_id
    ).first()
    
    if not favorite:
        raise HTTPException(status_code=404, detail="Favorite not found")
    
    db.delete(favorite)
    db.commit()
    return {"message": "Item removed from favorites"} 