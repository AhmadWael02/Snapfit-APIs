from fastapi import HTTPException, status, Depends, APIRouter
from sqlalchemy.orm import Session
import sqlalchemy  # For catching IntegrityError
import schemas
import models
import utils
from database import get_db
import oauth  # Assuming oauth.py and get_current_user are correctly defined
import traceback  # For detailed error logging

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/consumer-create", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def create_consumer(user: schemas.ConsumerCreate, db: Session = Depends(get_db)):
    try:
        # Check if the email already exists
        existing_user_by_email = db.query(models.User).filter(models.User.email == user.email).first()
        if existing_user_by_email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        # Check if the username already exists (assuming models.User.user_name and schemas.ConsumerCreate.username)
        existing_user_by_username = db.query(models.User).filter(models.User.user_name == user.username).first()
        if existing_user_by_username:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")

        # Hash the password
        hashed_password = utils.hash(user.password)

        # Create new User
        new_user = models.User(
            email=user.email,
            password=hashed_password,  # Assign hashed password here
            user_name=user.username  # Assuming your Pydantic schema 'user' has 'username'
        )
        db.add(new_user)
        db.flush()  # Flush to get the new_user.id for the foreign key relationship

        # Create new Consumer
        new_consumer = models.Consumer(
            consumer_id=new_user.id,
            size=user.size,
            gender=user.gender
        )
        db.add(new_consumer)

        db.commit()  # Commit both user and consumer in one transaction

        db.refresh(new_user)
        db.refresh(new_consumer)  # Good practice to refresh related object too

        return new_user

    except HTTPException as e:
        db.rollback()  # Rollback if an HTTPException we explicitly raised occurs
        raise e

    except sqlalchemy.exc.IntegrityError as e:  # Catch specific database integrity errors
        db.rollback()
        # You might want to parse e.orig to give a more specific error
        # For now, a generic one based on common integrity issues
        error_detail = "A database integrity error occurred. This could be due to a duplicate entry or a foreign key constraint."
        if "UNIQUE constraint failed: users.user_name" in str(e.orig):
            error_detail = "Username already taken."
        elif "UNIQUE constraint failed: users.email" in str(e.orig):
            error_detail = "Email already registered."
        print(f"IntegrityError: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_detail)

    except Exception as e:
        db.rollback()  # Undo any changes on unexpected error
        print(f"An unexpected error occurred in create_consumer: {e}")  # Log the actual error
        traceback.print_exc()  # Log the full traceback
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Something went wrong during sign-up.")


@router.post("/brand-create", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def create_brand(user: schemas.BrandCreate, db: Session = Depends(get_db)):
    try:
        # Check if the email already exists
        existing_user_by_email = db.query(models.User).filter(models.User.email == user.email).first()
        if existing_user_by_email:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        # Check if the username already exists (assuming models.User.user_name and schemas.BrandCreate.username)
        existing_user_by_username = db.query(models.User).filter(models.User.user_name == user.username).first()
        if existing_user_by_username:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")

        # Hash the password
        hashed_password = utils.hash(user.password)

        # Create a new User instance
        new_user = models.User(
            email=user.email,
            password=hashed_password,  # Assign hashed password here
            user_name=user.username  # Assuming your Pydantic schema 'user' has 'username'
        )
        db.add(new_user)
        db.flush()  # Flush to get the new_user.id

        # Create a new Brand instance
        new_brand = models.Brand(
            brand_id=new_user.id,
            description=user.description
        )
        db.add(new_brand)

        db.commit()  # Commit both user and brand in one transaction

        db.refresh(new_user)
        db.refresh(new_brand)

        return new_user

    except HTTPException as e:
        db.rollback()
        raise e

    except sqlalchemy.exc.IntegrityError as e:
        db.rollback()
        error_detail = "A database integrity error occurred during brand creation."
        if "UNIQUE constraint failed: users.user_name" in str(e.orig):
            error_detail = "Username already taken."
        elif "UNIQUE constraint failed: users.email" in str(e.orig):
            error_detail = "Email already registered."
        print(f"IntegrityError: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error_detail)

    except Exception as e:
        db.rollback()
        print(f"An unexpected error occurred in create_brand: {e}")  # Log the actual error
        traceback.print_exc()  # Log the full traceback
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Something went wrong during brand creation.")


@router.get("/{id}", response_model=schemas.UserInfo)  # Assuming UserInfo is defined in schemas
def get_user(
        id: int,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(oauth.get_current_user)  # Ensure oauth.get_current_user is correct
):
    try:
        # Authorization: Check if the current user is requesting their own data
        # Note: current_user is an instance of models.User, so it should have an 'id' attribute.
        if current_user.id != id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not authorized to access this user's data"
            )

        user = db.query(models.User).filter(models.User.id == id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with id: {id} not found"
            )
        # Ensure schemas.UserInfo is configured with from_attributes=True (Pydantic v2) or orm_mode=True (Pydantic v1)
        # and handles any field name differences (e.g., user_name vs username) if necessary.
        return user

    except HTTPException as e:
        # Re-raise HTTPExceptions so FastAPI handles them
        raise e

    except Exception as e:
        # Log unexpected errors for debugging
        print(f"An unexpected error occurred in get_user: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,  # Use descriptive constant
            detail="Something went wrong while retrieving the user."
        )