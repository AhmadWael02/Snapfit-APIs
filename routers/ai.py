from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Form, Body, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import numpy as np
from clip_utils import get_clip_embedding
from rembg import remove
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

import models as db_models
import oauth
from database import get_db

router = APIRouter(prefix="/ai", tags=["ai"])

# Load model once at startup
# MODEL_PATH = r'D:\College\Grad Project\Github Repos\models\subCategory_99%_resnet101.pth'
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
topwear_model.load_state_dict(torch.load(TOPWEAR_MODEL_PATH, map_location='cpu', weights_only=True))
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
bottomwear_model.load_state_dict(torch.load(BOTTOMWEAR_MODEL_PATH, map_location='cpu', weights_only=True))
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
shoes_model.load_state_dict(torch.load(SHOES_MODEL_PATH, map_location='cpu', weights_only=True))
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
occasion_model.load_state_dict(torch.load(OCCASION_MODEL_PATH, map_location='cpu', weights_only=True))
occasion_model.eval()
occasion_class_names = ['Casual', 'Formal', 'Sports']
occasion_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Subcategory classification model setup (main classifier)
SUBCATEGORY_MODEL_PATH = r'D:\snapfit_v1\snapfit_v1\assets\models\subCategory_classification (1).pth'
subcategory_model = models.resnet101(weights=None)
num_ftrs = subcategory_model.fc.in_features
subcategory_model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_ftrs, 5)
)
subcategory_model.load_state_dict(torch.load(SUBCATEGORY_MODEL_PATH, map_location='cpu', weights_only=True))
subcategory_model.eval()
subcategory_class_names = ['Bottomwear', 'Dress', 'Handbags', 'Shoes', 'Topwear']
subcategory_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MatchRequest(BaseModel):
    item_id: int


@router.post("/match")
def check_match(req: MatchRequest, db: Session = Depends(get_db),
                current_user: db_models.User = Depends(oauth.get_current_user)):
    # Very naive demo: 50-50 chance of match
    is_match = random.choice([True, False])
    return {"message": "Matches your style!" if is_match else "Might not fit your style"}


@router.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = subcategory_preprocess(img).unsqueeze(0)

        # First, classify the main category
        with torch.no_grad():
            output = subcategory_model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            main_category = subcategory_class_names[torch.argmax(probabilities).item()]

        # Based on the main category, call the specific classifier
        if main_category == 'Topwear':
            # Use topwear classifier
            img_tensor = topwear_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = topwear_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                subcategory = topwear_class_names[torch.argmax(probabilities).item()]

                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Upper Body",
                "subcategory": subcategory,
                "occasion": occasion
            }

        elif main_category == 'Bottomwear':
            # Use bottomwear classifier
            img_tensor = bottomwear_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = bottomwear_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                subcategory = bottomwear_class_names[torch.argmax(probabilities).item()]

                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Lower Body",
                "subcategory": subcategory,
                "occasion": occasion
            }

        elif main_category == 'Dress':
            # For Dress, return the main category and let frontend handle subcategory
            img_tensor = occasion_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Dress",
                "subcategory": "Dress",  # Let user select specific type
                "occasion": occasion
            }

        elif main_category == 'Handbags':
            # For Handbags, return Bags category and let frontend handle subcategory
            img_tensor = occasion_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Bags",
                "subcategory": "Bags",  # Let user select specific type
                "occasion": occasion
            }

        elif main_category == 'Shoes':
            # Use shoes classifier
            img_tensor = shoes_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                outputs = shoes_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                subcategory = shoes_class_names[torch.argmax(probabilities).item()]

                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Shoes",
                "subcategory": subcategory,
                "occasion": occasion
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_masked_image_from_db(item_id: int, db: Session, current_user: db_models.User):
    """Load masked image from database for a specific clothes item"""
    clothes_item = db.query(db_models.Clothes).filter(
        db_models.Clothes.id == item_id,
        db_models.Clothes.owner_id == current_user.id
    ).first()

    if not clothes_item:
        raise HTTPException(status_code=404, detail="Clothes item not found")

    # Construct the full path to the masked image
    masked_image_path = os.path.join("static", clothes_item.path)

    if not os.path.exists(masked_image_path):
        raise HTTPException(status_code=404, detail="Masked image not found")

    # Load and return the masked image
    img = Image.open(masked_image_path).convert('RGB')
    return img


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


@router.post("/classify-auto")
async def classify_auto(
        item_id: int = Query(..., description="ID of the clothes item to classify"),
        db: Session = Depends(get_db),
        current_user: db_models.User = Depends(oauth.get_current_user)
):
    try:
        # Load the masked image from database
        img = load_masked_image_from_db(item_id, db, current_user)
        img_tensor = subcategory_preprocess(img).unsqueeze(0)

        # First, classify the main category
        with torch.no_grad():
            output = subcategory_model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            main_category = subcategory_class_names[torch.argmax(probabilities).item()]

        # Based on the main category, call the specific classifier
        if main_category == 'Topwear':
            # Use topwear classifier
            img_tensor = topwear_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = topwear_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                subcategory = topwear_class_names[torch.argmax(probabilities).item()]

                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Upper Body",
                "subcategory": subcategory,
                "occasion": occasion
            }

        elif main_category == 'Bottomwear':
            # Use bottomwear classifier
            img_tensor = bottomwear_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                output = bottomwear_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                subcategory = bottomwear_class_names[torch.argmax(probabilities).item()]

                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Lower Body",
                "subcategory": subcategory,
                "occasion": occasion
            }

        elif main_category == 'Dress':
            # For Dress, return the main category and let frontend handle subcategory
            img_tensor = occasion_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Dress",
                "subcategory": "Dress",  # Let user select specific type
                "occasion": occasion
            }

        elif main_category == 'Handbags':
            # For Handbags, return Bags category and let frontend handle subcategory
            img_tensor = occasion_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Bags",
                "subcategory": "Bags",  # Let user select specific type
                "occasion": occasion
            }

        elif main_category == 'Shoes':
            # Use shoes classifier
            img_tensor = shoes_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                outputs = shoes_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                subcategory = shoes_class_names[torch.argmax(probabilities).item()]

                occasion_output = occasion_model(img_tensor)
                occasion_probabilities = torch.nn.functional.softmax(occasion_output[0], dim=0)
                occasion = occasion_class_names[torch.argmax(occasion_probabilities).item()]

            return {
                "category": "Shoes",
                "subcategory": subcategory,
                "occasion": occasion
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def metadata_compatibility(input_row, candidate_row):
    score = 0.0
    def safe_lower(val):
        return str(val).strip().lower() if val is not None else ""
    input_color = safe_lower(input_row.get('color'))
    candidate_color = safe_lower(candidate_row.get('color'))
    input_usage = safe_lower(input_row.get('occasion'))
    candidate_usage = safe_lower(candidate_row.get('occasion'))
    input_season = safe_lower(input_row.get('season'))
    candidate_season = safe_lower(candidate_row.get('season'))
    color_match = {
        'black': ['white', 'red', 'beige', 'pink', 'blue', 'green', 'yellow'],
        'white': ['black', 'blue', 'red', 'pink', 'grey', 'beige', 'green', 'brown'],
        'grey': ['black', 'white', 'pink', 'blue', 'red', 'yellow', 'green'],
        'blue': ['white', 'grey', 'beige', 'yellow', 'pink', 'brown'],
        'red': ['white', 'black', 'grey', 'beige', 'pink', 'brown'],
        'beige': ['white', 'black', 'blue', 'brown', 'green', 'pink'],
        'pink': ['white', 'grey', 'beige', 'blue', 'red'],
        'green': ['white', 'black', 'brown', 'beige', 'yellow', 'grey'],
        'brown': ['beige', 'white', 'green', 'blue', 'pink'],
        'yellow': ['blue', 'white', 'grey', 'green', 'brown']
    }
    if candidate_color in color_match.get(input_color, []):
        score += 0.4
    else:
        score += 0.2
    if input_usage == candidate_usage:
        score += 0.3
    if (
        input_season == candidate_season or
        input_season == "all season" or
        candidate_season == "all season"
    ):
        score += 0.3
    return score


def map_apparel_type(apparel_type, subtype=None):
    lower = apparel_type.lower()
    sub = (subtype or '').lower()
    # Topwear
    if lower in ['top', 'tops', 'upper body', 'jacket', 'jackets', 'shirt', 'shirts', 'sweater', 'sweaters', 'tshirt', 'tshirts'] or \
       sub in ['top', 'tops', 'jacket', 'jackets', 'shirt', 'shirts', 'sweater', 'sweaters', 'tshirt', 'tshirts']:
        return 'top'
    # Bottomwear
    if lower in ['bottom', 'bottoms', 'lower body', 'jeans', 'pants', 'trousers', 'shorts', 'skirts', 'track pants'] or \
       sub in ['jeans', 'pants', 'trousers', 'shorts', 'skirts', 'track pants', 'bottom', 'bottoms']:
        return 'bottom'
    # Shoes (expanded mapping)
    shoe_types = [
        'shoes', 'sneakers', 'sandals', 'boots', 'casual - formal shoes', 'sports shoes',
        'loafers', 'slippers', 'flip flops', 'heels', 'flats', 'moccasins', 'oxfords', 'derby', 'brogues', 'espadrilles', 'clogs', 'mules', 'platforms', 'wedges', 'trainers', 'running shoes', 'dress shoes', 'athletic shoes', 'canvas shoes', 'court shoes', 'ankle boots', 'chelsea boots', 'combat boots', 'desert boots', 'chukka boots', 'work boots', 'hiking boots', 'ballet flats', 'pumps', 'peep toes', 'mary janes', 'boat shoes', 'driving shoes', 'monk strap', 'wingtips', 'high tops', 'skate shoes', 'cross trainers', 'track shoes', 'soccer shoes', 'football boots', 'basketball shoes', 'tennis shoes', 'golf shoes', 'cleats', 'slides', 'espadrille', 'platform', 'wedge', 'sandal', 'boot', 'loafer', 'flat', 'heel', 'flip flop', 'slipper', 'moccasin', 'oxford', 'derby', 'brogue', 'espadrille', 'clog', 'mule', 'pump', 'peep toe', 'mary jane', 'boat shoe', 'driving shoe', 'monk strap', 'wingtip', 'high top', 'skate shoe', 'cross trainer', 'track shoe', 'soccer shoe', 'football boot', 'basketball shoe', 'tennis shoe', 'golf shoe', 'cleat', 'slide'
    ]
    if lower in shoe_types or sub in shoe_types:
        return 'shoes'
    return lower


@router.post("/recommend-outfit")
async def recommend_outfit(
    image: UploadFile = File(...),
    user_id: int = Form(...),
    db: Session = Depends(get_db),
    request: Request = None
):
    # 1. Mask the image
    image_bytes = await image.read()
    masked_bytes = remove(image_bytes)
    masked_img = Image.open(io.BytesIO(masked_bytes)).convert("RGB")

    # 2. Classify the masked image (same logic as classify_image)
    img_tensor = subcategory_preprocess(masked_img).unsqueeze(0)
    with torch.no_grad():
        output = subcategory_model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        main_category = subcategory_class_names[torch.argmax(probabilities).item()]

    # Map main_category to recommendation logic
    # If Topwear, recommend bottomwear and shoes; if Bottomwear, recommend topwear and shoes; if Shoes, recommend topwear and bottomwear
    if main_category == 'Topwear':
        recommend_cats = ['bottom', 'shoes']
    elif main_category == 'Bottomwear':
        recommend_cats = ['top', 'shoes']
    elif main_category == 'Shoes':
        recommend_cats = ['top', 'bottom']
    else:
        recommend_cats = ['top', 'bottom', 'shoes']

    # 3. Save masked image to a temporary file for embedding
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        masked_img.save(tmp, format='PNG')
        tmp_path = tmp.name
    try:
        embedding = np.array(get_clip_embedding(tmp_path)).reshape(1, -1)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    # 4. Get user gender
    consumer = db.query(db_models.Consumer).filter(db_models.Consumer.consumer_id == user_id).first()
    gender = consumer.gender.lower() if consumer and consumer.gender else 'male'
    # 5. Get shop and closet items
    shop_items = db.query(db_models.Clothes).join(db_models.Brand, db_models.Clothes.owner_id == db_models.Brand.brand_id).all()
    closet_items = db.query(db_models.Clothes).filter(db_models.Clothes.owner_id == user_id).all()
    print(f"USER_ID: {user_id}, CLOSET ITEMS FOUND: {[item.id for item in closet_items]}")
    # 6. Prepare candidates
    candidates = []
    for item in shop_items + closet_items:
        # Debug: print closet item info
        if item in closet_items:
            print(f"CLOSET ITEM: id={item.id}, apparel_type={item.apparel_type}, subtype={item.subtype}, gender={item.gender}, male_embedding={'set' if item.male_embedding else 'None'}, female_embedding={'set' if item.female_embedding else 'None'}")
        item_emb = item.male_embedding if gender == 'male' else item.female_embedding
        if item_emb is None:
            if item in closet_items:
                print(f"SKIP CLOSET ITEM {item.id}: No embedding for gender {gender}")
            continue
        try:
            arr = eval(item_emb)
            if isinstance(arr, set):
                arr = list(arr)
            item_emb = np.array(arr).reshape(1, -1)
        except Exception as e:
            if item in closet_items:
                print(f"SKIP CLOSET ITEM {item.id}: Embedding parse error: {e}")
            continue
        meta_score = metadata_compatibility(
            {'color': None, 'occasion': None, 'season': None},  # TODO: fill from classified image
            {'color': item.color, 'occasion': item.occasion, 'season': item.season}
        )
        visual_sim = cosine_similarity(embedding, item_emb)[0][0]
        final_score = 0.7 * visual_sim + 0.3 * meta_score
        # Fix image URL
        image_url = item.path
        if request:
            base_url = str(request.base_url).rstrip('/')
            image_url = f"{base_url}/static/{image_url.lstrip('/')}"
        else:
            if not image_url.startswith('http'):
                image_url = f"http://localhost:8000/static/{image_url.lstrip('/')}"
        candidates.append({
            'id': item.id,
            'path': image_url,
            'category': item.apparel_type,
            'subcategory': getattr(item, 'subtype', ''),
            'name': getattr(item, 'name', getattr(item, 'subtype', '')),
            'price': item.price,
            'brand': getattr(item, 'brand', None),
            'color': item.color,
            'size': item.size,
            'occasion': item.occasion,
            'gender': item.gender,
            'score': final_score,
            'source': 'shop' if item in shop_items else 'closet',
        })
    # 7. Recommend top items for each category
    recommendations = {}
    for cat in recommend_cats:
        cat_items = [c for c in candidates if map_apparel_type(c['category'], getattr(c, 'subcategory', None) if hasattr(c, 'subcategory') else None) == cat]
        closet_items = [c for c in cat_items if c['source'] == 'closet']
        shop_items = [c for c in cat_items if c['source'] == 'shop']
        best_closet = max(closet_items, key=lambda x: x['score']) if closet_items else None
        best_shop = max(shop_items, key=lambda x: x['score']) if shop_items else None
        recommendations[cat] = {
            'closet': best_closet,
            'shop': best_shop
        }
    print('AI Stylist Recommendations:', recommendations)
    return {'recommendations': recommendations, 'predicted_category': main_category}

# --- Save Outfit Endpoint ---
@router.post("/save-outfit")
async def save_outfit(
    user_id: int = Body(...),
    top_id: int = Body(None),
    bottom_id: int = Body(None),
    shoes_id: int = Body(None),
    name: str = Body("AI Stylist Outfit"),
    tags: list = Body([]),
    is_favorite: bool = Body(False),
    db: Session = Depends(get_db)
):
    # Only save if all required IDs are present
    if not (user_id and top_id and bottom_id and shoes_id):
        raise HTTPException(status_code=400, detail="Missing required outfit item IDs.")
    outfit = db_models.Outfit(
        user_id=user_id,
        top_id=top_id,
        bottom_id=bottom_id,
        shoes_id=shoes_id,
        name=name,
        tags=tags,
        is_favorite=is_favorite
    )
    db.add(outfit)
    db.commit()
    db.refresh(outfit)
    return {"message": "Outfit saved successfully", "outfit_id": outfit.id} 