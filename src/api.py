from fastapi import FastAPI
from schemas import EfficiencyPayload, DayIndexPayload
from inference import predict_data, predict_eff, get_model

app = FastAPI(title="Predict Heat Efficiency API")


@app.post("/predict/day", response_model=DayIndexPayload)
def predict_day_by_efficiency(data: EfficiencyPayload):
    day = predict_data(get_model(), data.efficiency)
    return DayIndexPayload(day=day)


@app.post("/predict/efficiency", response_model=EfficiencyPayload)
def predict_efficiency_by_day(data: DayIndexPayload):
    efficiency = predict_eff(get_model(), data.day)
    return EfficiencyPayload(efficiency=efficiency)
