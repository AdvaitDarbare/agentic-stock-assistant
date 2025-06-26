# route_schema.py
from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    step: Literal["sql_agent", "news_agent", "fallback"] = Field(
        ..., description="Decide which agent should handle the request."
    )