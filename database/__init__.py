# planner_app/database/__init__.py

from .connection import SessionLocal, engine
from .models import Base

# Create all tables if they do not exist
def init_db():
    Base.metadata.create_all(bind=engine)
