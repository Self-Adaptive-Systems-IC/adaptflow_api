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

    def json(self):
        return {
            "key": self.key,
            "model": self.model,
            "acc": self.accuracy,
            "cross_val": self.cross_val,
            "roc_score": self.roc_score,
            "pickle": self.pickle,
            "api": self.api,
        }

    def get_metrics(self):
        return {
            "key": self.key,
            "model": self.model,
            "acc": self.accuracy,
            "cross_val": self.cross_val,
            "roc_score": self.roc_score,
        }
