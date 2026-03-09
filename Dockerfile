FROM python:3.13-slim-bookworm

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY run_codegen.py ./
COPY proto ./proto
RUN uv run python run_codegen.py

COPY src ./src

EXPOSE 5000 50051 9090

RUN uv run python -c "from src.model import ToxicityModel; ToxicityModel().load()"

CMD ["sh", "-c", "uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 5000 & uv run python -m src.grpc_api & wait"]
