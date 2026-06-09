import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

load_dotenv()

# pool_pre_ping=True: checks connection is alive before using it
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# all models inherit from Base so SQLAlchemy knows which classes map to DB tables
Base = declarative_base()


# atomic transaction: commit on success, rollback on exception, always close the connection
@contextmanager
def transaction() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
