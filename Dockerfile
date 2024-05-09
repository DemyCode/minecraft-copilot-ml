FROM python:3.10

WORKDIR /minecraft-copilot-ml

RUN pip install poetry --no-cache-dir

COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-root --only main

COPY minecraft_copilot_ml minecraft_copilot_ml
COPY README.md .
RUN poetry install --only main

ENTRYPOINT ["./.venv/bin/python", "minecraft_copilot_ml"]
