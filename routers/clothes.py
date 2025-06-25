from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import shutil, os, traceback
import subprocess
from datetime import datetime
import requests
from rembg import remove
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models as tv_models, transforms
import models as db_models  # database models
import schemas
from database import get_db
import oauth
from utils import save_upload_file

router = APIRouter(prefix="/clothes", tags=["clothes"])

# ---- Helper ----

def _serialize_clothes(item: db_models.Clothes) -> schemas.Clothes:
    """Return a Pydantic schema instance for a single Clothes row."""
    return schemas.Clothes.from_orm(item)

# ---- CRUD ----

@router.get("/user", response_model=List[schemas.Clothes])
def get_user_clothes(db: Session = Depends(get_db), current_user: db_models.User = Depends(oauth.get_current_user)):
    try:
        clothes = db.query(db_models.Clothes).filter(db_models.Clothes.owner_id == current_user.id).all()
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
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(oauth.get_current_user)
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
        new_item = db_models.Clothes(
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
def update_clothes(item_id: int, updated: schemas.Clothes, db: Session = Depends(get_db), current_user: db_models.User = Depends(oauth.get_current_user)):
    item_query = db.query(db_models.Clothes).filter(db_models.Clothes.id == item_id, db_models.Clothes.owner_id == current_user.id)
    item = item_query.first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    item_query.update(updated.dict(exclude={'id'}), synchronize_session=False)
    db.commit()
    db.refresh(item)
    return item

@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_clothes(item_id: int, db: Session = Depends(get_db), current_user: db_models.User = Depends(oauth.get_current_user)):
    deleted = db.query(db_models.Clothes).filter(db_models.Clothes.id == item_id, db_models.Clothes.owner_id == current_user.id).delete(synchronize_session=False)
    db.commit()
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return

@router.post("/test-add-item", response_model=schemas.Clothes)
def test_add_item(db: Session = Depends(get_db), current_user: db_models.User = Depends(oauth.get_current_user)):
    item = db_models.Clothes(
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
async def upload_clothes_image(image: UploadFile = File(...), db: Session = Depends(get_db), current_user: db_models.User = Depends(oauth.get_current_user)):
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
        new_item = db_models.Clothes(
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
        
        # --- Automatic Subcategory Classification ---
        try:
            # Import torch and torchvision for model loading
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            from PIL import Image
            
            # Subcategory classification model setup
            SUBCATEGORY_MODEL_PATH = r'../snapfit_v1/assets/models/subCategory_classification.pth'
            subcategory_model = tv_models.resnet101(weights=None)
            num_ftrs = subcategory_model.fc.in_features
            subcategory_model.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(num_ftrs, 3)
            )
            subcategory_model.load_state_dict(torch.load(SUBCATEGORY_MODEL_PATH, map_location='cpu'))
            subcategory_model.eval()
            subcategory_class_names = ['Bottomwear', 'Shoes', 'Topwear']
            subcategory_preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Load the masked image and classify
            img = Image.open(static_masked_path).convert('RGB')
            img_tensor = subcategory_preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                output = subcategory_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                subcategory_prediction = subcategory_class_names[torch.argmax(probabilities).item()]
                
                # Map subcategory prediction to category
                category_mapping = {
                    'Topwear': 'Upper Body',
                    'Bottomwear': 'Lower Body', 
                    'Shoes': 'Shoes'
                }
                category = category_mapping.get(subcategory_prediction, 'Unknown')
            
            classification_result = {
                "subcategory": subcategory_prediction,
                "category": category
            }
        except Exception as e:
            print(f"Classification failed: {e}")
            classification_result = {
                "subcategory": "Unknown",
                "category": "Unknown"
            }
        
        # Return the ID, full URL path, and classification results for the frontend
        return {
            "id": new_item.id, 
            "path": masked_relative_path,
            "url": f"/static/{masked_relative_path}",
            "classification": classification_result
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Upload failed") 