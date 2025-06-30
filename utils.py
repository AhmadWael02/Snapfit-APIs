from passlib.context import CryptContext
import os
import shutil
from fastapi import UploadFile
import uuid
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config

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
    return os.path.join(folder, filename).replace("\\", "/")

def send_feedback_email(feedback: str, user_email: str, user_name: str = None) -> bool:
    """
    Send feedback email to yasfar2004@gmail.com using Gmail SMTP.
    
    Args:
        feedback: The feedback message
        user_email: Email of the user who sent feedback
        user_name: Name of the user (optional)
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Email configuration from config file
        sender_email = config.EMAIL_SENDER
        sender_password = config.EMAIL_PASSWORD
        receiver_email = config.EMAIL_RECEIVER
        
        # Check if email credentials are configured
        if sender_email == "your-email@gmail.com" or sender_password == "your-app-password":
            print("Warning: Email credentials not configured. Please set EMAIL_SENDER and EMAIL_PASSWORD in .env file")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"Snapfit App Feedback - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Email body
        body = f"""
        New feedback received from Snapfit App:
        
        User: {user_name or 'Unknown'} ({user_email})
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Feedback:
        {feedback}
        
        ---
        This is an automated message from the Snapfit App feedback system.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        
        print(f"Feedback email sent successfully to {receiver_email}")
        return True
        
    except Exception as e:
        print(f"Error sending feedback email: {e}")
        return False