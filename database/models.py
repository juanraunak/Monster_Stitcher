# planner_app/database/models.py

from sqlalchemy import Column, Integer, String, JSON, DateTime, func
from .connection import Base

class UserPlan(Base):
    __tablename__ = "monster_sticher"
    __table_args__ = {"schema": "mynoted_clone"}  # ✅ Specify the schema

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    plan_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

