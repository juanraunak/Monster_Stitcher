# planner_app/database/models.py

from sqlalchemy import Column, Integer, String, JSON, DateTime, func
from .connection import Base

class UserPlan(Base):
    __tablename__ = "monster_sticher"  # Correct table name

    id = Column(Integer, primary_key=True, index=True)  # Primary key
    user_id = Column(String, nullable=False)  # Assuming you want this field
    plan_json = Column(JSON, nullable=False)  # This field holds the plan data
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Auto-generated timestamp
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()  # Auto-updated on row update
    )
