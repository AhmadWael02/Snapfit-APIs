from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import shutil, os, traceback
import subprocess
from datetime import datetime
import requests
import sys
from rembg import remove
from PIL import Image
import io

import models, schemas
from database import get_db
import oauth
from utils import save_upload_file

router = APIRouter(prefix="/clothes", tags=["clothes"])

# ---- Helper ----

def _serialize_clothes(item: models.Clothes) -> schemas.Clothes:
    """Return a Pydantic schema instance for a single Clothes row."""
    return schemas.Clothes.from_orm(item)

# ---- CRUD ----

@router.get("/user", response_model=List[schemas.Clothes])
def get_user_clothes(db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    try:
        clothes = db.query(models.Clothes).filter(models.Clothes.owner_id == current_user.id).all()
        return clothes
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to fetch clothes")

PARSING_REPO = os.path.abspath("Self-Correction-Human-Parsing")
MODEL_PATH = os.path.join(PARSING_REPO, "models", "exp-schp-201908301523-atr.pth")
MODEL_URL = "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/model/exp-schp-201908301523-atr.pth"

# Ensure the repo and model are present
if not os.path.exists(PARSING_REPO):
    subprocess.run(["git", "clone", "https://github.com/PeikeLi/Self-Correction-Human-Parsing", PARSING_REPO], check=True)
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

@router.post("/", response_model=schemas.Clothes, status_code=status.HTTP_201_CREATED)
async def create_clothes(
    apparel_type: str = Form(...),
    subtype: str = Form(...),
    color: str = Form(...),
    size: str = Form(...),
    occasion: str = Form(""),
    brand: str = Form(""),
    gender: str = Form("Unisex"),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(oauth.get_current_user)
):
    try:
        # Save the image and get the relative path
        relative_path = save_upload_file(image, "images/clothes", f"user_{current_user.id}")
        abs_image_path = os.path.abspath(os.path.join("static", relative_path))
        image_id = os.path.splitext(os.path.basename(abs_image_path))[0]
        # --- Background Removal with rembg ---
        static_masked_dir = os.path.join("static", "images", "clothes", "masked")
        os.makedirs(static_masked_dir, exist_ok=True)
        masked_filename = f"masked_{image_id}.png"
        static_masked_path = os.path.join(static_masked_dir, masked_filename)
        with open(abs_image_path, "rb") as inp_file:
            input_data = inp_file.read()
        output_data = remove(input_data)
        with open(static_masked_path, "wb") as out_file:
            out_file.write(output_data)
        masked_relative_path = os.path.relpath(static_masked_path, "static")
        # Save the item with the masked image path
        new_item = models.Clothes(
            owner_id=current_user.id,
            apparel_type=apparel_type,
            subtype=subtype,
            color=color,
            size=size,
            occasion=occasion,
            gender=gender,
            path=masked_relative_path,
            purchase_link=None,
            price=None
        )
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        return new_item
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to create clothes item")

@router.put("/{item_id}", response_model=schemas.Clothes)
def update_clothes(item_id: int, updated: schemas.Clothes, db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    item_query = db.query(models.Clothes).filter(models.Clothes.id == item_id, models.Clothes.owner_id == current_user.id)
    item = item_query.first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    item_query.update(updated.dict(exclude={'id'}), synchronize_session=False)
    db.commit()
    db.refresh(item)
    return item

@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_clothes(item_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    deleted = db.query(models.Clothes).filter(models.Clothes.id == item_id, models.Clothes.owner_id == current_user.id).delete(synchronize_session=False)
    db.commit()
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return

@router.post("/test-add-item", response_model=schemas.Clothes)
def test_add_item(db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    item = models.Clothes(
        owner_id=current_user.id,
        gender="Unisex",
        apparel_type="Jacket",
        subtype="Bomber",
        color="Black",
        occasion="Casual",
        size="M",
        path="https://example.com/image.jpg",
        purchase_link="https://example.com/buy",
        price=99.99
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

# ---- Upload (image only) ----

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_clothes_image(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: models.User = Depends(oauth.get_current_user)):
    try:
        # Save the file and get the relative path
        relative_path = save_upload_file(file, "images/clothes", f"user_{current_user.id}")
        
        # Create a new clothes record
        new_item = models.Clothes(
            owner_id=current_user.id,
            path=relative_path,  # Store the relative path in the database
            apparel_type="Unknown",
            subtype="Unknown",
        )
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        
        # Return the ID and full URL path for the frontend
        return {
            "id": new_item.id, 
            "path": relative_path,
            "url": f"/static/{relative_path}"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Upload failed") 