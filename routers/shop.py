from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import traceback

import models
from database import get_db

router = APIRouter(prefix="/shop", tags=["shop"])

def _serialize(item: models.Clothes, user_name: str):
    return {
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
        "user_name": user_name,
    }

@router.get("/items")
def get_all_items(db: Session = Depends(get_db)):
    try:
        # Only include items where owner_id is a brand_id in the brand table
        brand_ids = db.query(models.Brand.brand_id).subquery()
        rows = (
            db.query(models.Clothes, models.User.user_name)
            .join(models.User, models.Clothes.owner_id == models.User.id)
            .filter(models.Clothes.owner_id.in_(brand_ids))
            .all()
        )
        return [_serialize(item, user_name) for item, user_name in rows]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to fetch items")

@router.get("/categories/{category}/items")
def get_items_by_category(category: str, db: Session = Depends(get_db)):
    rows = (
        db.query(models.Clothes, models.User.user_name)
        .join(models.User, models.Clothes.owner_id == models.User.id)
        .filter(models.Clothes.apparel_type.ilike(category))
        .all()
    )
    return [_serialize(item, user_name) for item, user_name in rows]

@router.get("/brands/{brand_id}/items")
def get_items_by_brand(brand_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(models.Clothes, models.User.user_name)
        .join(models.User, models.Clothes.owner_id == models.User.id)
        .filter(models.Clothes.owner_id == brand_id)
        .all()
    )
    return [_serialize(item, user_name) for item, user_name in rows]

@router.get("/items/{item_id}")
def get_shop_item_by_id(item_id: int, db: Session = Depends(get_db)):
    try:
        # Only include items where owner_id is a brand_id in the brand table
        brand_ids = db.query(models.Brand.brand_id).subquery()
        row = (
            db.query(models.Clothes, models.User.user_name)
            .join(models.User, models.Clothes.owner_id == models.User.id)
            .filter(models.Clothes.id == item_id)
            .filter(models.Clothes.owner_id.in_(brand_ids))
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Shop item not found")
        item, user_name = row
        return _serialize(item, user_name)
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to fetch item") 