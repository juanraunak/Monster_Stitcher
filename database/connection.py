# planner_app/database/connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = (
    "postgresql://monsteradmin:M0nsteradmin@monsterdb.postgres.database.azure.com:5432/postgres?sslmode=require"
)

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for model declarations
Base = declarative_base()
