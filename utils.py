from passlib.context import CryptContext
import os
import shutil
from fastapi import UploadFile
import uuid
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash(password: str) -> str:
    return pwd_context.hash(password)

def verify(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def save_upload_file(upload_file: UploadFile, folder: str, prefix: str = "") -> str:
    """
    Save an uploaded file to the specified folder with a unique filename.
    Returns the file path relative to the static directory.
    """
    # Ensure directory exists
    full_folder_path = os.path.join("static", folder)
    os.makedirs(full_folder_path, exist_ok=True)
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}_{upload_file.filename}"
    
    # Create the full file path
    file_path = os.path.join(full_folder_path, filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # Return the relative path for database storage
    return os.path.join(folder, filename)