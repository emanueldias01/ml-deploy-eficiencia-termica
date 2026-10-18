from pydantic import BaseModel, Field


class DayIndexPayload(BaseModel):
    day: int = Field(..., ge=1)


class EfficiencyPayload(BaseModel):
    efficiency: float = Field(..., ge=0, le=100)

class Model(BaseModel):
    name: str
