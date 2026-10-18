from fastapi import FastAPI
from schemas import EfficiencyPayload, DayIndexPayload, Model
from inference import predict_data, predict_eff, get_model, get_models

app = FastAPI(title="Predict Heat Efficiency API")


@app.post("/predict/day", response_model=DayIndexPayload)
def predict_day_by_efficiency(data: EfficiencyPayload):
    day = predict_data(get_model(), data.efficiency)
    return DayIndexPayload(day=day)


@app.post("/predict/efficiency", response_model=EfficiencyPayload)
def predict_efficiency_by_day(data: DayIndexPayload):
    efficiency = predict_eff(get_model(), data.day)
    return EfficiencyPayload(efficiency=efficiency)

@app.get("/models", response_model=list[Model])
def get_saved_models():
    return get_models()