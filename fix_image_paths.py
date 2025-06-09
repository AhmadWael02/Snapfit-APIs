"""
Script to migrate existing images to the new static file structure.
Run this script once after setting up the new static file serving.
"""

import os
import shutil
from sqlalchemy.orm import Session
from database import engine, get_db
import models

def ensure_directories():
    """Create necessary directories for static file serving"""
    os.makedirs("static/images/clothes", exist_ok=True)
    os.makedirs("static/images/profile_pics", exist_ok=True)

def migrate_images():
    """Move existing images to the new static directory structure and update database paths"""
    # Get database session
    db = next(get_db())
    
    try:
        print("Starting image migration...")
        
        # Migrate clothes images
        clothes_items = db.query(models.Clothes).all()
        migrated_clothes = 0
        
        for item in clothes_items:
            if not item.path:
                continue
                
            # Skip already migrated items
            if item.path.startswith("images/"):
                continue
                
            old_path = item.path
            
            # Only process if the file exists
            if os.path.exists(old_path):
                # Extract filename
                filename = os.path.basename(old_path)
                
                # Create new path
                new_rel_path = f"images/clothes/user_{item.owner_id}_{filename}"
                new_full_path = os.path.join("static", new_rel_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
                
                # Copy the file
                try:
                    shutil.copy2(old_path, new_full_path)
                    
                    # Update database
                    item.path = new_rel_path
                    migrated_clothes += 1
                    print(f"Migrated clothes image: {old_path} -> {new_rel_path}")
                except Exception as e:
                    print(f"Error copying {old_path}: {e}")
        
        # Migrate profile pictures
        users = db.query(models.User).all()
        migrated_profiles = 0
        
        for user in users:
            if not user.profile_picture:
                continue
                
            # Skip already migrated items
            if user.profile_picture.startswith("images/"):
                continue
                
            old_path = user.profile_picture
            
            # Only process if the file exists
            if os.path.exists(old_path):
                # Extract filename
                filename = os.path.basename(old_path)
                
                # Create new path
                new_rel_path = f"images/profile_pics/user_{user.id}_{filename}"
                new_full_path = os.path.join("static", new_rel_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
                
                # Copy the file
                try:
                    shutil.copy2(old_path, new_full_path)
                    
                    # Update database
                    user.profile_picture = new_rel_path
                    migrated_profiles += 1
                    print(f"Migrated profile image: {old_path} -> {new_rel_path}")
                except Exception as e:
                    print(f"Error copying {old_path}: {e}")
        
        # Commit all changes
        db.commit()
        print(f"Migration complete. Migrated {migrated_clothes} clothes images and {migrated_profiles} profile pictures.")
        
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    ensure_directories()
    migrate_images() 