from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

import models as db_models
import oauth
from database import get_db
from torchvision import models as tv_models

router = APIRouter(prefix="/ai", tags=["ai"])

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

# Topwear model setup
TOPWEAR_MODEL_PATH = r'../snapfit_v1/assets/models/Topwear_93.3%_resnet101.pth'
topwear_model = models.resnet101(weights=None)
num_ftrs = topwear_model.fc.in_features
topwear_model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_ftrs, 5)
)
topwear_model.load_state_dict(torch.load(TOPWEAR_MODEL_PATH, map_location='cpu'))
topwear_model.eval()
topwear_class_names = ['Jackets', 'Shirts', 'Sweaters', 'Tops', 'Tshirts']
topwear_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Bottomwear model setup
BOTTOMWEAR_MODEL_PATH = r'../snapfit_v1/assets/models/Bottomwear_94.8%_resnet101.pth'
bottomwear_model = models.resnet101(weights=None)
num_ftrs = bottomwear_model.fc.in_features
bottomwear_model.fc = nn.Linear(num_ftrs, 5)
bottomwear_model.load_state_dict(torch.load(BOTTOMWEAR_MODEL_PATH, map_location='cpu'))
bottomwear_model.eval()
bottomwear_class_names = ['Jeans', 'Shorts', 'Skirts', 'Track Pants', 'Trousers']
bottomwear_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Shoes model setup
SHOES_MODEL_PATH = r'../snapfit_v1/assets/models/Shoes_91%_resnet50.pth'
shoes_model = models.resnet152(weights=None)
num_ftrs = shoes_model.fc.in_features
shoes_model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_ftrs, 3)
)
shoes_model.load_state_dict(torch.load(SHOES_MODEL_PATH, map_location='cpu'))
shoes_model.eval()
shoes_class_names = ['Casual - Formal Shoes', 'Sandals', 'Sports Shoes']
shoes_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Occasion model setup
OCCASION_MODEL_PATH = r'../snapfit_v1/assets/models/usage2.pth'
occasion_model = models.resnet152(weights=None)
num_ftrs = occasion_model.fc.in_features
occasion_model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_ftrs, 3)
)
occasion_model.load_state_dict(torch.load(OCCASION_MODEL_PATH, map_location='cpu'))
occasion_model.eval()
occasion_class_names = ['Casual', 'Formal', 'Sports']
occasion_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MatchRequest(BaseModel):
    item_id: int

@router.post("/match")
def check_match(req: MatchRequest, db: Session = Depends(get_db), current_user: db_models.User = Depends(oauth.get_current_user)):
    # Very naive demo: 50-50 chance of match
    is_match = random.choice([True, False])
    return {"message": "Matches your style!" if is_match else "Might not fit your style"}

@router.post("/classify-image")
def classify_image(file: UploadFile = File(...)):
    # Model temporarily unavailable. Always return a default response.
    return {"class": "Unknown"}

def load_masked_image_from_db(item_id: int, db: Session, current_user: db_models.User) -> Image.Image:
    """Load the masked image from database for a specific item."""
    item = db.query(db_models.Clothes).filter(
        db_models.Clothes.id == item_id, 
        db_models.Clothes.owner_id == current_user.id
    ).first()
    
    if not item or not item.path:
        raise HTTPException(status_code=404, detail="Item not found or no image path")
    
    # Construct the full path to the masked image
    image_path = os.path.join("static", item.path)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return Image.open(image_path).convert('RGB')

@router.post("/classify-subcategory")
async def classify_subcategory(
    item_id: int = Query(..., description="ID of the clothes item to classify"),
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(oauth.get_current_user)
):
    try:
        # Load the masked image from database
        img = load_masked_image_from_db(item_id, db, current_user)
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
        
        return {
            "subcategory": subcategory_prediction,
            "category": category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-topwear")
async def classify_topwear(
    item_id: int = Query(..., description="ID of the clothes item to classify"),
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(oauth.get_current_user)
):
    try:
        # Load the masked image from database
        img = load_masked_image_from_db(item_id, db, current_user)
        img_tensor = topwear_preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = topwear_model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            topwear_prediction = topwear_class_names[torch.argmax(probabilities).item()]
        
        with torch.no_grad():
            occasion_output = occasion_model(img_tensor)
            occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
            occasion_prediction = occasion_class_names[torch.argmax(occasion_probabilities).item()]
        
        return {
            "topwear": topwear_prediction,
            "occasion": occasion_prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-bottomwear")
async def classify_bottomwear(
    item_id: int = Query(..., description="ID of the clothes item to classify"),
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(oauth.get_current_user)
):
    try:
        # Load the masked image from database
        img = load_masked_image_from_db(item_id, db, current_user)
        img_tensor = bottomwear_preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = bottomwear_model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            bottomwear_prediction = bottomwear_class_names[torch.argmax(probabilities).item()]
        
        with torch.no_grad():
            occasion_output = occasion_model(img_tensor)
            occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
            occasion_prediction = occasion_class_names[torch.argmax(occasion_probabilities).item()]
        
        return {
            "bottomwear": bottomwear_prediction,
            "occasion": occasion_prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-shoes")
async def classify_shoes(
    item_id: int = Query(..., description="ID of the clothes item to classify"),
    db: Session = Depends(get_db),
    current_user: db_models.User = Depends(oauth.get_current_user)
):
    try:
        # Load the masked image from database
        img = load_masked_image_from_db(item_id, db, current_user)
        img_tensor = shoes_preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = shoes_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            shoes_prediction = shoes_class_names[torch.argmax(probabilities).item()]
        
        with torch.no_grad():
            occasion_output = occasion_model(img_tensor)
            occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
            occasion_prediction = occasion_class_names[torch.argmax(occasion_probabilities).item()]
        
        return {
            "shoes": shoes_prediction,
            "occasion": occasion_prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 