import joblib
from src.utils.logger import get_logger
import pandas as pd
import argparse
from pathlib import Path
import os
from functools import lru_cache

logger = get_logger(__name__)


def load_model(path: str) -> joblib:
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_model():
    path = get_latest_model()
    return load_model(path)


def predict_eff(model, day_index: int) -> float:
    logger.info(f"Predicting heat efficiency for day index: {day_index}")
    df = pd.DataFrame({"day_index": [day_index]})
    predict = model.predict(df)[0]
    logger.info(f"Predicted heat efficiency: {predict}")
    return predict


def predict_data(model, y) -> int:
    logger.info(f"Predicting heat efficiency for dataset")
    coef = model.coef_
    intercept = model.intercept_
    logger.info(f"Model coefficients: {coef}, intercept: {intercept}")
    day = (y - intercept) / coef
    day = round(float(day[0]))
    logger.info(f"Predicted day index for heat efficiency {y}: {day}")
    return day


def get_latest_model() -> Path:
    artifacts_dir = Path("artifacts")

    models = list(artifacts_dir.glob("linear_regression_model-*.pkl"))

    if not models:
        raise FileNotFoundError("No trained models found in artifacts/")

    latest_model = max(models, key=lambda path: path.name)

    logger.info(f"Latest model found: {latest_model}")

    return latest_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--day-index", type=int)
    parser.add_argument("--heat-efficiency", type=float)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models is not None:
        models = os.listdir("artifacts")

        if models:
            print("Available models:")
            for model in models:
                print(model)
        else:
            print("No models found in artifacts/")

        pass

    if args.model_path is None:
        path = get_latest_model()
        model = load_model(path)
    else:
        model = load_model(args.model_path)

    if args.day_index is not None:
        predict_eff(model, day_index=args.day_index)

    if args.heat_efficiency is not None:
        predict_data(model, y=args.heat_efficiency)
