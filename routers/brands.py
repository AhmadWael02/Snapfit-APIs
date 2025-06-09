from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

import models, schemas
from database import get_db

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