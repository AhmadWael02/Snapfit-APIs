"""
Script to download sample clothing images for testing.
This will download images from placeholder services and save them to the static directory.
"""

import os
import requests
import random
from database import get_db
import models

# Categories for sample clothing
CATEGORIES = {
    "tops": ["tshirt", "blouse", "sweater", "hoodie", "shirt"],
    "bottoms": ["jeans", "pants", "skirt", "shorts"],
    "dress": ["casual_dress", "formal_dress", "party_dress"],
    "bags": ["backpack", "clutch", "tote"],
    "shoes": ["sneakers", "boots", "sandals", "heels"]
}

# Colors for variety
COLORS = ["red", "blue", "green", "black", "white", "yellow", "purple", "orange", "pink", "brown"]

def download_image(url, save_path):
    """Download an image from a URL and save it to the specified path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def create_sample_images(num_images=20):
    """Create sample images for the application"""
    # Ensure directories exist
    os.makedirs("static/images/clothes", exist_ok=True)

    # Get database session
    db = next(get_db())

    try:
        print(f"Downloading {num_images} sample images...")

        for i in range(num_images):
            # Randomly select category and subcategory
            category = random.choice(list(CATEGORIES.keys()))
            subcategory = random.choice(CATEGORIES[category])
            color = random.choice(COLORS)

            # Create a unique filename
            filename = f"sample_{category}_{subcategory}_{color}_{i+1}.jpg"
            save_rel_path = f"images/clothes/{filename}"
            save_full_path = os.path.join("static", save_rel_path)

            # Generate a random image from placeholder service
            width = random.randint(400, 800)
            height = random.randint(400, 800)
            image_url = f"https://picsum.photos/{width}/{height}"

            # Download the image
            if download_image(image_url, save_full_path):
                print(f"Downloaded: {filename}")

                # Create a database entry for the image
                apparel_type = "Top" if category == "tops" else "Bottom" if category == "bottoms" else "Dress" if category == "dress" else "Accessory" if category == "bags" else "Footwear"
                
                # Get a random user ID (assuming you have users in the database)
                user = db.query(models.User).first()
                if not user:
                    print("No users found in the database. Skipping database entry.")
                    continue
                
                # Create clothes entry
                new_item = models.Clothes(
                    owner_id=user.id,
                    path=save_rel_path,
                    apparel_type=apparel_type,
                    subtype=subcategory.capitalize(),
                    color=color.capitalize(),
                    gender="Unisex",
                    occasion="Casual"
                )
                db.add(new_item)
        
        # Commit all changes
        db.commit()
        print(f"Sample images created successfully.")
        
    except Exception as e:
        db.rollback()
        print(f"Error creating sample images: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_sample_images() 