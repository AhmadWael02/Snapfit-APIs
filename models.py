from database import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP, text, ForeignKey, JSON, Float, Boolean
from sqlalchemy.orm import relationship

#Hello! This is a sample code for a SQLAlchemy ORM model for a clothing recommendation system.
# The model includes classes for User, Brand, Consumer, Store, Clothes, and Recommendation.

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    user_name = Column(String, nullable=False, unique=True)
    phone = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    brand = relationship("Brand", uselist=False, back_populates="user")
    consumer = relationship("Consumer", uselist=False, back_populates="user")
    storage = relationship("Store", uselist=False, back_populates="user")
    clothes = relationship("Clothes", back_populates="owner")
    recommendation = relationship("Recommendation", uselist=False, back_populates="user")
    likes = relationship("UserLikes", back_populates="user")
    profile_picture = Column(String, nullable=True)
    preferences = relationship("UserPreferences", uselist=False, back_populates="user")
    favorites = relationship("UserFavorites", back_populates="user")
    outfits = relationship("Outfit", back_populates="user")


class Brand(Base):
    __tablename__ = "brand"
    brand_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    description = Column(String, nullable=False)
    user = relationship("User", back_populates="brand")


class Consumer(Base):
    __tablename__ = "consumer"
    consumer_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    size = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    user = relationship("User", back_populates="consumer")


class Store(Base):
    __tablename__ = "store"
    user_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    number_of_clothes = Column(Integer, nullable=False)
    clothes = Column(JSON, nullable=False)
    user = relationship("User", back_populates="storage")


class Clothes(Base):
    __tablename__ = "clothes"
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    gender = Column(String, nullable=False)
    apparel_type = Column(String, nullable=False)
    subtype = Column(String, nullable=False)
    color = Column(String, nullable=False)
    occasion = Column(String, nullable=False)
    size = Column(String, nullable=False)
    path = Column(String, nullable=False)
    purchase_link = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    season = Column(String, nullable=False, default='All Year Long')
    male_embedding = Column(String, nullable=True)
    female_embedding = Column(String, nullable=True)
    path_3d = Column(String, nullable=True)
    owner = relationship("User", back_populates="clothes")
    favorited_by = relationship("UserFavorites", back_populates="item")
    outfits_as_top = relationship("Outfit", foreign_keys="Outfit.top_id", back_populates="top")
    outfits_as_bottom = relationship("Outfit", foreign_keys="Outfit.bottom_id", back_populates="bottom")
    outfits_as_shoes = relationship("Outfit", foreign_keys="Outfit.shoes_id", back_populates="shoes")


class Recommendation(Base):
    __tablename__ = "recommendation"
    user_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    recommendation = Column(JSON, nullable=False)
    user = relationship("User", back_populates="recommendation")


class UserLikes(Base):
    __tablename__ = "user_likes"
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    item_id = Column(Integer, ForeignKey('clothes.id'), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    user = relationship("User", back_populates="likes")


class UserPreferences(Base):
    __tablename__ = "user_preferences"
    user_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    fit_preference = Column(String, nullable=False, default="Regular")
    lifestyle_preferences = Column(JSON, nullable=False, default=list)
    season_preference = Column(String, nullable=False, default="Auto")
    age_group = Column(String, nullable=False, default="18-24")
    preferred_colors = Column(JSON, nullable=False, default=list)
    excluded_categories = Column(JSON, nullable=False, default=list)
    user = relationship("User", back_populates="preferences")


class UserFavorites(Base):
    __tablename__ = "user_favorites"
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    item_id = Column(Integer, ForeignKey('clothes.id'), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    user = relationship("User", back_populates="favorites")
    item = relationship("Clothes", back_populates="favorited_by")


class Outfit(Base):
    __tablename__ = "outfits"
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    top_id = Column(Integer, ForeignKey('clothes.id'), nullable=False)
    bottom_id = Column(Integer, ForeignKey('clothes.id'), nullable=False)
    shoes_id = Column(Integer, ForeignKey('clothes.id'), nullable=False)
    bags = Column(Integer, ForeignKey('clothes.id'), nullable=True)
    name = Column(String, nullable=True)
    tags = Column(JSON, nullable=False, default=list)
    is_favorite = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    
    user = relationship("User", back_populates="outfits")
    top = relationship("Clothes", foreign_keys=[top_id], back_populates="outfits_as_top")
    bottom = relationship("Clothes", foreign_keys=[bottom_id], back_populates="outfits_as_bottom")
    shoes = relationship("Clothes", foreign_keys=[shoes_id], back_populates="outfits_as_shoes")


class ItemEvent(Base):
    __tablename__ = "item_events"
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("clothes.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"))
    event_type = Column(String(32), nullable=False)  # 'item_click', 'visit_store', 'recommendation'
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))