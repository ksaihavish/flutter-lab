# Flutter Lab web app + API (serving only: NumPy inference, no PyTorch).
#   docker build -t flutter-lab .
#   docker run -p 8000:8000 --env-file .env flutter-lab     # .env optional (Claude supervisor)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLUTTER_AI_HOME=/app

WORKDIR /app
RUN useradd --create-home --uid 1000 app

# Dependencies first (better layer caching), then the code and assets
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[llm]"

COPY frontend ./frontend
COPY models ./models
COPY data ./data
RUN mkdir -p outputs && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["python", "-m", "flutter_ai", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
