from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Result(Base):
    __tablename__ = "Result"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, index=True)
    model = Column(String)
    accuracy = Column(Float)
    cross_val = Column(Float)
    roc_score = Column(Float)
    pickle = Column(String)
    api = Column(String)
