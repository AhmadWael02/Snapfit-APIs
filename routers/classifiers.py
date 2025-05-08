from fastapi import HTTPException, status, Depends, APIRouter, Form
from sqlalchemy.orm import Session
import schemas
import models
from database import get_db
from PIL import Image
import io
from fastapi import UploadFile, File
from predictor import predict_bottom_type
from oauth import get_current_user

router = APIRouter(prefix="/classifiers", tags=["classifiers"])


@router.post("/predict_bottom", status_code=status.HTTP_201_CREATED, response_model=schemas.ClassifierResponse)
def predict_bottom(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)  # ⬅ gets user from Bearer token
):
    try:
        # Read the uploaded image
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))

        # Predict subtype using your model
        subtype_prediction = predict_bottom_type(image)

        # Add the prediction result to the database
        new_clothes = models.Clothes(
            owner_id=current_user.id,  # ⬅ use user ID from token
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
