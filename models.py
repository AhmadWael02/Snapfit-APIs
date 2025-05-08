from database import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP, text, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship

#Hello! This is a sample code for a SQLAlchemy ORM model for a clothing recommendation system.
# The model includes classes for User, Brand, Consumer, Store, Clothes, and Recommendation.

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, nullable=False, index=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    user_name = Column(String, nullable=False, unique=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    brand = relationship("Brand", uselist=False, back_populates="user")
    consumer = relationship("Consumer", uselist=False, back_populates="user")
    storage = relationship("Store", uselist=False, back_populates="user")
    clothes = relationship("Clothes", back_populates="owner")
    recommendation = relationship("Recommendation", uselist=False, back_populates="user")


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
    gender = Column(String, nullable=True)
    apparel_type = Column(String, nullable=True)
    subtype = Column(String, nullable=True)
    color = Column(String, nullable=True)
    occasion = Column(String, nullable=True)
    size = Column(String, nullable=True)
    path = Column(String, nullable=True)
    purchase_link = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    owner = relationship("User", back_populates="clothes")


class Recommendation(Base):
    __tablename__ = "recommendation"
    user_id = Column(Integer, ForeignKey('user.id'), primary_key=True)
    recommendation = Column(JSON, nullable=False)
    user = relationship("User", back_populates="recommendation")