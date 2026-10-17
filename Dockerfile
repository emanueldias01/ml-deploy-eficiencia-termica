FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src /app/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.14-slim AS runtimer

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY data /app/data

RUN mkdir -p artifacts

ENV PATH="/app/.venv/bin:$PATH"
ENV DB_PATH="data/heat_exchanger.db"
ENV MODEL_PATH="artifacts/linear_regression_model.pkl"

CMD ["python" , "src/train.py"]

## if you have run inference, run docker run <image-name:tag> sh -c "python src/train.py && python src/inference.py --day-index=<number of days> --heat-efficiency=<percentage of efficiency>"