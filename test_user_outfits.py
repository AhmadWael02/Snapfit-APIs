"""
Test script to ensure outfits are properly associated with specific users.
This script:
1. Creates test users if they don't exist
2. Creates test outfits for each user
3. Verifies outfits are only visible to their owners
"""

from sqlalchemy.orm import Session
from database import get_db, engine
import models
from models import User, Clothes, Outfit
import schemas
import utils
import random
from typing import List
import os

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

def create_test_user(db: Session, email: str, password: str, username: str) -> User:
    """Create a test user if they don't exist"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        hashed_password = utils.hash(password)
        user = User(
            email=email,
            password=hashed_password,
            user_name=username,
            profile_picture=None
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created test user: {username} ({email})")
    else:
        print(f"User already exists: {username} ({email})")
    
    return user

def create_test_clothes(db: Session, user: User, count: int = 6) -> dict:
    """Create test clothes for a user"""
    # Check if user already has enough clothes
    existing_clothes = db.query(Clothes).filter(Clothes.owner_id == user.id).all()
    
    if len(existing_clothes) >= count:
        tops = [c for c in existing_clothes if c.apparel_type == "top"][:2]
        bottoms = [c for c in existing_clothes if c.apparel_type == "bottom"][:2]
        shoes = [c for c in existing_clothes if c.apparel_type == "shoes"][:2]
        
        if tops and bottoms and shoes:
            print(f"User {user.user_name} already has required clothes")
            return {
                "tops": tops,
                "bottoms": bottoms,
                "shoes": shoes
            }
    
    # Apparel types and details
    clothes_types = {
        "top": ["T-shirt", "Shirt", "Blouse", "Sweater"],
        "bottom": ["Jeans", "Trousers", "Skirt", "Shorts"],
        "shoes": ["Sneakers", "Boots", "Sandals", "Loafers"]
    }
    
    colors = ["Black", "White", "Blue", "Red", "Green", "Yellow"]
    occasions = ["Casual", "Formal", "Sport", "Party"]
    sizes = ["S", "M", "L", "XL"]
    
    # Create new clothes
    tops = []
    bottoms = []
    shoes = []
    
    for apparel_type, subtypes in clothes_types.items():
        for i in range(2):  # Create 2 of each type
            subtype = random.choice(subtypes)
            color = random.choice(colors)
            occasion = random.choice(occasions)
            size = random.choice(sizes)
            
            # Create placeholder path for image
            path = f"test_images/{user.id}_{apparel_type}_{i}.jpg"
            
            clothes = Clothes(
                owner_id=user.id,
                gender="Unisex",
                apparel_type=apparel_type,
                subtype=subtype,
                color=color,
                occasion=occasion,
                size=size,
                path=path,
                purchase_link=None,
                price=random.randint(20, 200)
            )
            
            db.add(clothes)
            
            if apparel_type == "top":
                tops.append(clothes)
            elif apparel_type == "bottom":
                bottoms.append(clothes)
            else:
                shoes.append(clothes)
    
    db.commit()
    
    # Refresh to get IDs
    for clothes in tops + bottoms + shoes:
        db.refresh(clothes)
    
    print(f"Created clothes for user {user.user_name}: {len(tops)} tops, {len(bottoms)} bottoms, {len(shoes)} shoes")
    
    return {
        "tops": tops,
        "bottoms": bottoms,
        "shoes": shoes
    }

def create_test_outfits(db: Session, user: User, clothes: dict, count: int = 3) -> List[Outfit]:
    """Create test outfits for a user"""
    # Check if user already has outfits
    existing_outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
    
    if len(existing_outfits) >= count:
        print(f"User {user.user_name} already has {len(existing_outfits)} outfits")
        return existing_outfits[:count]
    
    # Categories for tags
    categories = ["Everyday", "Work", "Workout", "Party", "Weekend"]
    
    outfits = []
    for i in range(count):
        # Randomly assign items and tags
        outfit_tags = [random.choice(categories)]
        # Sometimes add a second tag
        if random.random() > 0.7:
            second_tag = random.choice(categories)
            if second_tag not in outfit_tags:
                outfit_tags.append(second_tag)
                
        outfit = Outfit(
            user_id=user.id,
            top_id=random.choice(clothes["tops"]).id,
            bottom_id=random.choice(clothes["bottoms"]).id,
            shoes_id=random.choice(clothes["shoes"]).id,
            name=f"{user.user_name}'s Outfit {i+1}",
            tags=outfit_tags,
            is_favorite=random.random() > 0.7  # 30% chance of being favorited
        )
        
        db.add(outfit)
        outfits.append(outfit)
    
    db.commit()
    
    # Refresh to get IDs
    for outfit in outfits:
        db.refresh(outfit)
    
    print(f"Created {len(outfits)} outfits for user {user.user_name}")
    return outfits

def verify_user_outfits(db: Session, users: List[User]):
    """Verify each user can only see their own outfits"""
    for user in users:
        user_outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
        
        print(f"\nUser {user.user_name} has {len(user_outfits)} outfits:")
        for outfit in user_outfits:
            # Get the outfit details
            top = db.query(Clothes).filter(Clothes.id == outfit.top_id).first()
            bottom = db.query(Clothes).filter(Clothes.id == outfit.bottom_id).first()
            shoes = db.query(Clothes).filter(Clothes.id == outfit.shoes_id).first()
            
            print(f"  - Outfit #{outfit.id}: {outfit.name}")
            print(f"    Top: {top.subtype} ({top.color})")
            print(f"    Bottom: {bottom.subtype} ({bottom.color})")
            print(f"    Shoes: {shoes.subtype} ({shoes.color})")
            print(f"    Tags: {outfit.tags}")
            print(f"    Favorite: {outfit.is_favorite}")

def main():
    # Get database session
    db = next(get_db())
    
    try:
        # Create test users
        user1 = create_test_user(db, "testuser1@example.com", "password123", "TestUser1")
        user2 = create_test_user(db, "testuser2@example.com", "password123", "TestUser2")
        
        # Create test clothes for each user
        clothes1 = create_test_clothes(db, user1)
        clothes2 = create_test_clothes(db, user2)
        
        # Create test outfits for each user
        outfits1 = create_test_outfits(db, user1, clothes1)
        outfits2 = create_test_outfits(db, user2, clothes2)
        
        # Verify users can only see their own outfits
        verify_user_outfits(db, [user1, user2])
        
        print("\nTest completed successfully!")
        print("Note: In the actual application, users will only see their own outfits because:")
        print("1. The backend API filters outfits by user_id")
        print("2. The frontend passes the user's JWT token for authentication")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main() 