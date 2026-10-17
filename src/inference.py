import joblib
from utils.logger import get_logger
import pandas as pd
import argparse

logger = get_logger(__name__)


def load_model(path: str) -> joblib:
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def predict_eff(model, day_index: int):
    logger.info(f"Predicting heat efficiency for day index: {day_index}")
    df = pd.DataFrame({"day_index": [day_index]})
    predict = model.predict(df)[0]
    logger.info(f"Predicted heat efficiency: {predict}")


def predict_data(model, y):
    logger.info(f"Predicting heat efficiency for dataset")
    coef = model.coef_
    intercept = model.intercept_
    logger.info(f"Model coefficients: {coef}, intercept: {intercept}")
    day = (y - intercept) / coef
    logger.info(f"Predicted day index for heat efficiency {y}: {day[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="artifacts/linear_regression_model.pkl")
    parser.add_argument("--day-index", type=int)
    parser.add_argument("--heat-efficiency", type=float)
    args = parser.parse_args()

    model = load_model(args.model_path)

    if args.day_index is not None:
        predict_eff(model, day_index=args.day_index)

    if args.heat_efficiency is not None:
        predict_data(model, y=args.heat_efficiency)
