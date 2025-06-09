from fastapi import HTTPException, status, Depends, APIRouter, Form, UploadFile, File
from sqlalchemy.orm import Session
import io
from PIL import Image
import sys
import os
from typing import List

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import modules from parent directory
import models
from database import get_db
from predictor import predict_bottom_type
from oauth import get_current_user
import schemas

# Create the schemas in this file if it doesn't exist in parent directory
class ClassifierResponse:
    def __init__(self, subtype: str):
        self.subtype = subtype

router = APIRouter(prefix="/classifiers", tags=["classifiers"])

@router.post("/predict_bottom", status_code=status.HTTP_201_CREATED)
def predict_bottom(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    try:
        # Read the uploaded image
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))

        # Predict subtype using your model
        subtype_prediction = predict_bottom_type(image)

        # Add the prediction result to the database
        new_clothes = models.Clothes(
            owner_id=current_user.id,
            subtype=subtype_prediction
        )
        db.add(new_clothes)
        db.commit()
        db.refresh(new_clothes)

        # Return the prediction result
        return {"subtype": subtype_prediction}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )

@router.get("/my_closet", response_model=List[schemas.Clothes])
def get_my_closet(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return db.query(models.Clothes).filter(models.Clothes.owner_id == current_user.id).all()

@router.put("/item/{item_id}", response_model=schemas.Clothes)
def edit_closet_item(
    item_id: int,
    item: schemas.Clothes,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_item = db.query(models.Clothes).filter(models.Clothes.id == item_id, models.Clothes.owner_id == current_user.id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    for field, value in item.dict(exclude_unset=True).items():
        setattr(db_item, field, value)
    db.commit()
    db.refresh(db_item)
    return db_item

@router.delete("/item/{item_id}", status_code=204)
def delete_closet_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_item = db.query(models.Clothes).filter(models.Clothes.id == item_id, models.Clothes.owner_id == current_user.id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(db_item)
    db.commit()
    return
