from sqlalchemy import create_engine, engine_from_config, Table, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from src.database.models.File import Base as BaseFile, File
from src.database.models.Result import Base as BaseResult, Result

DATABASE_URL = "sqlite:///./adapt.db"

engine = create_engine(DATABASE_URL)

Base = declarative_base()


def init_db():
    print("Initializaing the database...")
    Base.metadata.create_all(bind=engine)
    BaseFile.metadata.create_all(bind=engine)
    BaseResult.metadata.create_all(bind=engine)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
