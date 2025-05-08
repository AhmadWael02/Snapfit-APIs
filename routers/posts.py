from fastapi import HTTPException, status, Depends, APIRouter
from sqlalchemy.orm import Session
import schemas
import models
from database import get_db
import oauth
from typing import Optional

router = APIRouter(prefix="/posts", tags=["posts"])


@router.get("/", response_model=list[schemas.Post])
def get_posts(db: Session = Depends(get_db), get_current_user: str = Depends(oauth.get_current_user), limit: int = 10,search: Optional[str] = ""):
    #cursor.execute(""" SELECT * FROM posts """)
    #posts = cursor.fetchall()
    posts = db.query(models.post).filter(models.post.title.contains(search)).limit(limit).all()
    return posts

@router.post("/", status_code=status.HTTP_201_CREATED)
def create_post(post: schemas.PostCreate, db: Session = Depends(get_db), get_current_user: str = Depends(oauth.get_current_user)):

    #cursor.execute("""Insert into posts (title, body, age) values (%s, %s ,%s) returning * """,(post.title, post.body, post.age))
    #new_post = cursor.fetchone()
    #connection.commit()

    new_post = models.post(user_id = get_current_user.id,**post.dict())
    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    return new_post



@router.get("/{id}")
def get_post(id: int, db: Session = Depends(get_db), get_current_user: str = Depends(oauth.get_current_user)):
    #cursor.execute(""" SELECT * FROM posts WHERE id = %s returning * """,(id,))
    #post = cursor.fetchone()

    post = db.query(models.post).filter(models.post.id == id).first()

    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Post with id: {id} not found")
    
    return post


@router.delete("/{id}")
def delete_post(id: int, db: Session = Depends(get_db), get_current_user: str = Depends(oauth.get_current_user)):
    #cursor.execute(""" DELETE FROM posts WHERE id = %s returning * """,(id,))
    #delete_post=cursor.fetchone()
    #connection.commit()

    delete_post = db.query(models.post).filter(models.post.id == id).delete(synchronize_session=False)
    db.commit()
    if delete_post.first() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Post with id: {id} not found")
    
    if delete_post.user_id != get_current_user.id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="You are not authorized to delete this post")
    return {"Post Deleted"}



@router.put("/{id}")
def update_post(id: int, post: schemas.PostCreate,  db: Session = Depends(get_db), get_current_user: str = Depends(oauth.get_current_user)):
    #cursor.execute(""" UPDATE posts SET title = %s, body = %s, age = %s WHERE id = %s returning * """,(post.title, post.body, post.age, id))
    #update_post = cursor.fetchone()
    #connection.commit()

    update_post = db.query(models.post).filter(models.post.id == id)
    post = update_post.first()

    if post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Post with id: {id} not found")
    
    if post.user_id != get_current_user.id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="You are not authorized to update this post")
    
    update_post.update(post.dict(), synchronize_session=False)
    db.commit()
    return "Post updated"