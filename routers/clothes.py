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
from clip_utils import get_clip_embedding

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


@router.post("/", response_model=schemas.Clothes, status_code=status.HTTP_201_CREATED)
async def create_clothes(
        apparel_type: str = Form(...),
        subtype: str = Form(...),
        color: str = Form(...),
        size: str = Form(...),
        occasion: str = Form(""),
        brand: str = Form(""),
        gender: str = Form("Unisex"),
        season: str = Form(None),
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
        # --- Auto-fill season if not provided ---
        _sub = subtype.strip().lower()
        if not season or season.strip() == "":
            if _sub in ["tshirt", "top", "skirt", "short", "sandal"]:
                season = "Summer"
            elif _sub in ["sweaters", "casual jacket"]:
                season = "Winter"
            else:
                season = "All Year Long"
        # --- Compute CLIP embedding for masked image ---
        abs_masked_path = os.path.abspath(os.path.join("static", masked_relative_path))
        embedding = get_clip_embedding(abs_masked_path)
        # --- Determine gender from Consumer table ---
        consumer = db.query(models.Consumer).filter(models.Consumer.consumer_id == current_user.id).first()
        gender_db = consumer.gender.lower() if consumer and consumer.gender else "unisex"
        male_embedding = None
        female_embedding = None
        if gender_db == "male":
            male_embedding = embedding
        elif gender_db == "female":
            female_embedding = embedding
        # Save the item with the masked image path and embedding
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
            price=None,
            season=season,
            male_embedding=male_embedding,
            female_embedding=female_embedding
        )
        db.add(new_item)
        db.commit()
        db.refresh(new_item)
        return new_item
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to create clothes item")


@router.put("/{item_id}", response_model=schemas.Clothes)
def update_clothes(item_id: int, updated: schemas.Clothes, db: Session = Depends(get_db),
                   current_user: models.User = Depends(oauth.get_current_user)):
    item_query = db.query(models.Clothes).filter(models.Clothes.id == item_id,
                                                 models.Clothes.owner_id == current_user.id)
    item = item_query.first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    item_query.update(updated.dict(exclude={'id'}), synchronize_session=False)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_clothes(item_id: int, db: Session = Depends(get_db),
                   current_user: models.User = Depends(oauth.get_current_user)):
    deleted = db.query(models.Clothes).filter(models.Clothes.id == item_id,
                                              models.Clothes.owner_id == current_user.id).delete(
        synchronize_session=False)
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

@router.post("/upload-image-only", status_code=status.HTTP_201_CREATED)
async def upload_image_only(image: UploadFile = File(...), current_user: models.User = Depends(oauth.get_current_user)):
    try:
        # Save the file and get the relative path
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

        # Return only the masked image URL (no database entry created)
        return {
            "url": f"/static/{masked_relative_path}"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Upload failed")


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_clothes_image(image: UploadFile = File(...), db: Session = Depends(get_db),
                               current_user: models.User = Depends(oauth.get_current_user)):
    try:
        # Save the file and get the relative path
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

        # Create a new clothes record with the masked image path
        new_item = models.Clothes(
            owner_id=current_user.id,
            path=masked_relative_path,  # Store the masked image path in the database
            apparel_type="Unknown",
            subtype="Unknown",
            gender="Unisex",  # Default value for required field
            color="Unknown",  # Default value for required field
            occasion="Unknown",  # Default value for required field
            size="Unknown",  # Default value for required field
            purchase_link=None,
            price=None
        )
        db.add(new_item)
        db.commit()
        db.refresh(new_item)

        # Return the ID and full URL path for the frontend
        return {
            "id": new_item.id,
            "path": masked_relative_path,
            "url": f"/static/{masked_relative_path}"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Upload failed") 