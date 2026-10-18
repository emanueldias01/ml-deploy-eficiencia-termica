from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utils.logger import get_logger
from utils.versioning import get_version
from sqlalchemy import create_engine
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from dotenv import load_dotenv
import os
from pathlib import Path


logger = get_logger(__name__)


def read_dataset(path: str) -> pd.DataFrame:
    logger.info("read dataset")
    engine = create_engine(f"sqlite:///{path}")
    df = pd.read_sql_query(
        "SELECT timestamp, heat_efficiency FROM heat_exchanger ORDER BY timestamp",
        engine,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_index"] = (df["timestamp"] - df["timestamp"].min()).dt.days
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.2):
    logger.info("Split dataset into train and test sets")
    train_size = int(len(df) * (1 - test_size))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df


def train(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    logger.info("Train model")
    model = LinearRegression()
    model.fit(X, y)

    logger.info(
        f"Model trained with coefficients: {model.coef_} and intercept: {model.intercept_}"
    )
    return model


def evaluate_model(model: LinearRegression, X: pd.DataFrame, y: pd.Series):
    logger.info("Evaluate model")
    score = model.score(X, y)
    pred = model.predict(X)
    MAE = mean_absolute_error(y, pred)
    RMSE = root_mean_squared_error(y, pred)
    logger.info(f"Model R2 score: {score}")
    logger.info(f"Model MAE: {MAE}")
    logger.info(f"Model RMSE: {RMSE}")
    return score


def save_model(model: LinearRegression, path: str) -> None:
    logger.info(f"Save model to {path}")

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)


if __name__ == "__main__":
    load_dotenv()

    dataset_path = os.getenv("DB_PATH")
    model_path = os.getenv("MODEL_PATH")

    df = read_dataset(dataset_path)
    X = df[["day_index"]]
    y = df["heat_efficiency"]

    train_df, test_df = split_dataset(df, test_size=0.2)
    X_train = train_df[["day_index"]]
    y_train = train_df["heat_efficiency"]
    X_test = test_df[["day_index"]]
    y_test = test_df["heat_efficiency"]

    model = train(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    version = get_version()
    save_model(model, f"{model_path}{version}.pkl")
