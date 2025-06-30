"""
Script to fix items in the database that have full URLs stored instead of relative paths.
This fixes the issue where item.path contains full URLs like 'http://10.0.2.2:8000/static/images/clothes/jacket.jpeg'
"""

from sqlalchemy.orm import Session
from database import engine, get_db
import models
import re

def fix_url_paths():
    """Fix items that have full URLs stored instead of relative paths"""
    # Get database session
    db = next(get_db())
    
    try:
        print("Starting URL path fix...")
        
        # Find all clothes items
        clothes_items = db.query(models.Clothes).all()
        fixed_items = 0
        
        for item in clothes_items:
            if not item.path:
                continue
                
            # Check if the path is a full URL
            if item.path.startswith('http://') or item.path.startswith('https://'):
                print(f"Found item with full URL: {item.path}")
                
                # Extract the relative path from the URL
                # Pattern: http://host:port/static/images/clothes/filename
                match = re.search(r'/static/(.+)$', item.path)
                if match:
                    relative_path = match.group(1)
                    print(f"Extracting relative path: {relative_path}")
                    
                    # Update the database
                    item.path = relative_path
                    fixed_items += 1
                    print(f"Fixed item {item.id}: {item.path}")
                else:
                    print(f"Could not extract relative path from: {item.path}")
        
        # Commit all changes
        db.commit()
        print(f"URL path fix complete. Fixed {fixed_items} items.")
        
    except Exception as e:
        db.rollback()
        print(f"Error during URL path fix: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    fix_url_paths() 