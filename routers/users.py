from fastapi import HTTPException, status, Depends, APIRouter
from sqlalchemy.orm import Session
import schemas
import models
import utils
from database import get_db
import oauth

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/consumer-create", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def create_consumer(user: schemas.ConsumerCreate, db: Session = Depends(get_db)):
    try:
        # Check if the user already exists
        existing_user = db.query(models.User).filter(models.User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        # Hash the password
        hashed_password = utils.hash(user.password)
        user.password = hashed_password

        # Create new User
        new_user = models.User(
            email=user.email,
            password=user.password,
            user_name=user.username
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Create new Consumer
        new_consumer = models.Consumer(
            consumer_id=new_user.id,
            size=user.size,
            gender=user.gender
        )
        db.add(new_consumer)
        db.commit()
        db.refresh(new_consumer)

        return new_user

    except HTTPException as e:
        raise e  # Let FastAPI handle the HTTPException

    except Exception as e:
        db.rollback()  # Undo any changes on unexpected error
        raise HTTPException(status_code=500, detail="Something went wrong during sign-up.")


@router.post("/brand-create", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def create_brand(user: schemas.BrandCreate, db: Session = Depends(get_db)):
    try:
        # Check if the user already exists
        existing_user = db.query(models.User).filter(models.User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        # Hash the password
        hashed_password = utils.hash(user.password)
        user.password = hashed_password

        # Create a new User instance
        new_user = models.User(
            email=user.email,
            password=user.password,
            user_name=user.username
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Create a new Brand instance
        new_brand = models.Brand(
            brand_id=new_user.id,
            description=user.description
        )
        db.add(new_brand)
        db.commit()
        db.refresh(new_brand)

        return new_user

    except HTTPException as e:
        raise e  # Let FastAPI handle the HTTPException

    except Exception as e:
        db.rollback()  # Undo any changes on unexpected error
        raise HTTPException(status_code=500, detail="Something went wrong during brand creation.")



@router.get("/{id}", response_model=schemas.UserInfo)
def get_user(
    id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(oauth.get_current_user)
):
    try:
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

        return user

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while retrieving the user."
        )
