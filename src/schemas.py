from pydantic import BaseModel


class DayIndexPayload(BaseModel):
    day: int


class EfficiencyPayload(BaseModel):
    efficiency: float
