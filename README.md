# Thermal Exchange Tube Efficiency Prediction

This repository contains an end-to-end machine learning project for exploring heat-exchanger data, training a model, and serving heat-efficiency predictions. It includes an exploratory analysis notebook, a SQLite dataset, a training and inference workflow, a FastAPI service, Docker images, Docker Compose orchestration, and a GitHub Actions build-and-validation workflow.

## Project overview

The current model is a scikit-learn `LinearRegression` model. It uses elapsed time (`day_index`) as its only input feature to estimate `heat_efficiency`; the other temperature measurements in the dataset are available for analysis but are not currently model features.

Training reads the records in timestamp order, derives `day_index` from the earliest timestamp, uses the first 80% of records for training and the last 20% for testing, and logs the test-set R², MAE, and RMSE. Each trained model is saved as a timestamped joblib artifact. Inference can estimate efficiency from a day index or estimate a day index from an efficiency value.

The included SQLite database currently contains 175 observations. The model is a baseline based on the time trend in this dataset; predictions should not be treated as a validated thermodynamic simulation or as a substitute for engineering assessment.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/heat_exchanger.db` | SQLite dataset containing the heat-exchanger observations |
| `notebooks/eda.ipynb` | Exploratory data analysis, including summary statistics, missing-value checks, plots, and a linear trend |
| `src/train.py` | Loads and prepares the dataset, trains and evaluates the model, and saves a versioned artifact |
| `src/inference.py` | Loads the latest or a specified model and provides prediction helpers and a command-line interface |
| `src/api.py` | FastAPI endpoints for predictions and listing saved models |
| `src/schemas.py` | Pydantic request and response models |
| `src/utils/` | Logging and timestamp-based model version utilities |
| `Dockerfile.train` | Container image for model training |
| `Dockerfile.inference` | Container image for command-line inference |
| `Dockerfile.api` | Container image for the prediction API |
| `docker-compose.yml` | Runs training and then the API with a shared model-artifact volume |
| `.github/workflows/build.yml` | Builds and validates the training and inference Docker images on pushes to `main` |

## Dataset

The `heat_exchanger` table contains the following columns:

| Column | Description |
| --- | --- |
| `timestamp` | Observation date |
| `water_inlet_temperature` | Water inlet temperature |
| `glycol_inlet_temperature` | Glycol inlet temperature |
| `out_glycol_temperature` | Glycol outlet temperature |
| `out_water_temperature` | Water outlet temperature |
| `heat_efficiency` | Heat-exchanger efficiency used as the prediction target |

The training pipeline reads `timestamp` and `heat_efficiency`. It converts each timestamp into an integer day offset from the earliest observation and trains on that single derived feature.

## Requirements

- Python 3.14, as specified by `.python-version` and `pyproject.toml`
- [uv](https://docs.astral.sh/uv/) for installing the locked Python dependencies
- Docker and the Docker Compose plugin for containerized workflows

## Local setup

From the repository root, install the project dependencies:

```bash
uv sync
```

## Explore the data

Open `notebooks/eda.ipynb` in Jupyter and run it from the repository root. The notebook reads `data/heat_exchanger.db`, derives `day_index`, inspects missing values and descriptive statistics, plots efficiency over time, and compares a fitted linear trend with the observed values.

## Train a model

Set the database location and the prefix used for model artifact filenames, then run the training script:

```bash
export DB_PATH=data/heat_exchanger.db
export MODEL_PATH=artifacts/linear_regression_model-
uv run python src/train.py
```

Training writes a file such as `artifacts/linear_regression_model-2026-10-18_18-03-19.pkl`. The `artifacts/` directory is excluded from version control.

## Run inference from the command line

After training, predict efficiency for a day index using the latest model:

```bash
uv run python src/inference.py --day-index 10
```

Or estimate the day index for a specified efficiency:

```bash
uv run python src/inference.py --heat-efficiency 90
```

To use a specific artifact instead of the latest one:

```bash
uv run python src/inference.py \
  --model-path artifacts/linear_regression_model-2026-10-18_18-03-19.pkl \
  --day-index 10
```

List the available models with:

```bash
uv run python src/inference.py --list-models
```

The model must already exist in `artifacts/` or be provided through `--model-path`.

## Run the API

The API loads the latest model from `artifacts/`. Train a model first, then start the development server:

```bash
uv run uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Endpoints

| Method | Endpoint | Request | Response |
| --- | --- | --- | --- |
| `POST` | `/predict/efficiency` | `{"day": 10}` | `{"efficiency": 96.2}` |
| `POST` | `/predict/day` | `{"efficiency": 90}` | `{"day": 305}` |
| `GET` | `/models` | None | A list of saved model names |

`day` must be at least 1. `efficiency` must be between 0 and 100. Prediction values in the examples are illustrative.

Example request:

```bash
curl -X POST http://localhost:8000/predict/efficiency \
  -H "Content-Type: application/json" \
  -d '{"day": 10}'
```

## Run with Docker Compose

Docker Compose builds the training and API images, runs training first, and shares its model artifact with the API through a named volume:

```bash
docker compose up --build
```

The API is then available at [http://localhost:8000](http://localhost:8000), with interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs). Stop the services with `Ctrl+C`; remove the containers and the named artifact volume with `docker compose down -v`.

The individual images can also be built directly:

```bash
docker build -f Dockerfile.train -t heat-efficiency-train .
docker build -f Dockerfile.inference -t heat-efficiency-inference .
docker build -f Dockerfile.api -t heat-efficiency-api .
```

## Continuous integration

The GitHub Actions workflow runs on pushes to `main`. It builds the training and inference Docker images, transfers them between jobs as an artifact, runs training against the bundled dataset, checks that a model artifact was created, and runs an inference command against that artifact. The workflow does not currently publish images to a registry or deploy the API.
