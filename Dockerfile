FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (cached layer)
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY src/ src/
COPY config/ config/

# Install project
RUN uv sync --frozen --no-dev

FROM python:3.12-slim

WORKDIR /app

# Copy installed environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config

ENV PATH="/app/.venv/bin:$PATH"
ENV LITERAVORE_CONFIG=/app/config/default.yml
ENV LITERAVORE_DATA_DIR=/data

# Create a non-root user/group that matches the host owner (UID/GID 1000 by default).
# At runtime docker-compose passes --user $(id -u):$(id -g) so the actual UID/GID
# used is always the caller's, keeping volume-mounted files owned by the host user.
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -s /bin/sh appuser

VOLUME /data

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["literavore", "serve"]
