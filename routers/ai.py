from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
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

router = APIRouter(prefix="/ai", tags=["ai"])

# Load model once at startup
# MODEL_PATH = r'D:\snapfit_v1\snapfit_v1\assets\models\subCategory_99%_resnet101.pth'
# model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
# model.eval()
# class_names = ['Bottomwear', 'Dress', 'Shoes', 'Topwear']
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# Topwear model setup
TOPWEAR_MODEL_PATH = r'D:\snapfit_v1\snapfit_v1\assets\models\Topwear_93.3%_resnet101.pth'
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
BOTTOMWEAR_MODEL_PATH = r'D:\snapfit_v1\snapfit_v1\assets\models\Bottomwear_94.8%_resnet101.pth'
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
SHOES_MODEL_PATH = r'D:\snapfit_v1\snapfit_v1\assets\models\Shoes_91%_resnet50.pth'
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
OCCASION_MODEL_PATH = r'D:\snapfit_v1\snapfit_v1\assets\models\usage2.pth'
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

@router.post("/classify-topwear")
async def classify_topwear(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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
async def classify_bottomwear(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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
async def classify_shoes(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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