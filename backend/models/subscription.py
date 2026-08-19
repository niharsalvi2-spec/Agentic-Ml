from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum

class PlanType(str, Enum):
    PERSONAL = "personal"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class SubscriptionBase(BaseModel):
    user_id: str
    plan_type: PlanType
    is_active: bool = True

class SubscriptionCreate(SubscriptionBase):
    pass

class SubscriptionInDB(SubscriptionBase):
    id: str
    start_date: datetime
    end_date: Optional[datetime] = None
    
    class Config:
        from_attributes = True
