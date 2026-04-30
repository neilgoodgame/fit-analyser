FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Install dependencies (cached layer)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

# Copy source
COPY fit_analyser/ ./fit_analyser/

# Install the package itself
RUN poetry install --only main

ENTRYPOINT ["poetry", "run", "fit-analyser"]
