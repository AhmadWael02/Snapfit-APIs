import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

# Email Configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "your-email@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your-app-password")
EMAIL_RECEIVER = "yasfar2004@gmail.com"

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./snapfit.db")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Other configurations can be added here

class Settings(BaseSettings):
    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    database_username: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    hf_api_key: str
    serpapi_key: str
    class Config:
        env_file = ".env"

settings = Settings()