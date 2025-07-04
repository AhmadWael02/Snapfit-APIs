from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile
from sqlalchemy.orm import Session
from typing import List
import traceback
import os
from clip_utils import get_clip_embedding
from rembg import remove
from utils import save_upload_file
import json

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
    print(
        f"Received item event: item_id={event.item_id}, user_id={event.user_id}, event_type={event.event_type}")  # Debug log
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
        users_clicked = db.query(models.ItemEvent.user_id).filter(models.ItemEvent.item_id == item.id,
                                                                  models.ItemEvent.event_type == 'item_click').distinct().count()
        visit_store = db.query(models.ItemEvent.user_id).filter(models.ItemEvent.item_id == item.id,
                                                                models.ItemEvent.event_type == 'visit_store').distinct().count()
        recommended = db.query(models.ItemEvent).filter(models.ItemEvent.item_id == item.id,
                                                        models.ItemEvent.event_type == 'recommendation').count()
        stats.append(schemas.ItemEventStats(
            item_id=item.id,
            item_name=item.subtype or item.apparel_type,
            item_photo_url=item.path,
            users_clicked=users_clicked,
            visit_store=visit_store,
            recommended=recommended
        ))
    return stats


@router.post("/add-item", status_code=201)
async def add_brand_item(
        apparel_type: str = Form(...),
        subtype: str = Form(...),
        color: str = Form(...),
        size: str = Form(...),
        occasion: str = Form(""),
        gender: str = Form(...),  # Brand owner chooses the gender
        season: str = Form(None),
        price: float = Form(None),
        purchase_link: str = Form(None),
        image: UploadFile = File(...),
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)
):
    try:
        # Verify user is a brand
        brand = db.query(models.Brand).filter(models.Brand.brand_id == current_user.id).first()
        if not brand:
            raise HTTPException(status_code=403, detail="Only brand users can add items")

        # Save the image and get the relative path
        relative_path = save_upload_file(image, "images/clothes", f"user_{current_user.id}")
        abs_image_path = os.path.abspath(os.path.join("static", relative_path))
        image_id = os.path.splitext(os.path.basename(abs_image_path))[0]

        # --- Remove background removal for brand uploads ---
        # Use the original image path for embedding and storage
        # --- Auto-fill season if not provided ---
        _sub = subtype.strip().lower()
        if not season or season.strip() == "":
            if _sub in ["tshirt", "top", "skirt", "short", "sandal"]:
                season = "Summer"
            elif _sub in ["sweaters", "casual jacket"]:
                season = "Winter"
            else:
                season = "All Year Long"

        # --- Compute CLIP embedding for original image ---
        clip_embedding = get_clip_embedding(abs_image_path)
        vse_embedding = None
        try:
            from inception_utils import get_vse_embedding
            vse_embedding = get_vse_embedding(abs_image_path)
        except Exception:
            vse_embedding = None
        # --- Store embedding based on brand owner's choice ---
        gender_lower = gender.lower()
        male_embedding = None
        female_embedding = None
        if gender_lower == "male":
            male_embedding = json.dumps(clip_embedding)
        elif gender_lower == "female":
            male_embedding = json.dumps(clip_embedding)  # Always store CLIP embedding for female items
            female_embedding = json.dumps(vse_embedding) if vse_embedding is not None else None
        elif gender_lower == "unisex":
            male_embedding = json.dumps(clip_embedding)
            female_embedding = json.dumps(vse_embedding) if vse_embedding is not None else None

        # Save the item with the original image path and embedding
        new_item = models.Clothes(
            owner_id=current_user.id,
            apparel_type=apparel_type,
            subtype=subtype,
            color=color,
            size=size,
            occasion=occasion,
            gender=gender,
            path=relative_path,
            purchase_link=purchase_link,
            price=price,
            season=season,
            male_embedding=male_embedding,
            female_embedding=female_embedding
        )
        db.add(new_item)
        db.commit()
        db.refresh(new_item)

        # Build the full URL for the returned path (ensure forward slashes)
        backend_url = os.getenv('BACKEND_URL', 'http://10.0.2.2:8000')
        rel_path = relative_path.replace('\\', '/')
        full_url = f"{backend_url}/static/{rel_path}"

        return {
            "id": new_item.id,
            "path": full_url,
            "message": "Item added successfully"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unable to add brand item") 