from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

import models, schemas
from database import get_db
from oauth import get_current_user

router = APIRouter(prefix="/brands", tags=["brands"])

# List all brands with name and description
@router.get("/", response_model=List[dict])
def list_brands(db: Session = Depends(get_db)):
    brands = (
        db.query(models.Brand, models.User)
        .join(models.User, models.Brand.brand_id == models.User.id)
        .all()
    )
    return [
        {
            "brand_id": b.brand_id,
            "name": u.user_name,
            "description": b.description,
        }
        for b, u in brands
    ]

# Get single brand
@router.get("/{brand_id}")
def get_brand(brand_id: int, db: Session = Depends(get_db)):
    brand = db.query(models.Brand).filter(models.Brand.brand_id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    user = db.query(models.User).filter(models.User.id == brand_id).first()
    return {
        "brand_id": brand.brand_id,
        "name": user.user_name if user else None,
        "description": brand.description,
    }

@router.post("/item_event", status_code=201)
def log_item_event(event: schemas.ItemEventCreate, db: Session = Depends(get_db)):
    print(f"Received item event: item_id={event.item_id}, user_id={event.user_id}, event_type={event.event_type}")  # Debug log
    db_event = models.ItemEvent(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return {"status": "success"}

@router.get("/items/statistics", response_model=List[schemas.ItemEventStats])
def get_brand_item_statistics(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # Only allow brand users
    brand = db.query(models.Brand).filter(models.Brand.brand_id == current_user.id).first()
    if not brand:
        raise HTTPException(status_code=403, detail="Not a brand user")
    # Get all items owned by this brand
    items = db.query(models.Clothes).filter(models.Clothes.owner_id == current_user.id).all()
    stats = []
    for item in items:
        users_clicked = db.query(models.ItemEvent.user_id).filter(models.ItemEvent.item_id == item.id, models.ItemEvent.event_type == 'item_click').distinct().count()
        visit_store = db.query(models.ItemEvent.user_id).filter(models.ItemEvent.item_id == item.id, models.ItemEvent.event_type == 'visit_store').distinct().count()
        recommended = db.query(models.ItemEvent).filter(models.ItemEvent.item_id == item.id, models.ItemEvent.event_type == 'recommendation').count()
        stats.append(schemas.ItemEventStats(
            item_id=item.id,
            item_name=item.subtype or item.apparel_type,
            item_photo_url=item.path,
            users_clicked=users_clicked,
            visit_store=visit_store,
            recommended=recommended
        ))
    return stats 