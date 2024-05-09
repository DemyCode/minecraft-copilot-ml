FROM python:3.10

WORKDIR /minecraft-copilot-ml

RUN pip install poetry --no-cache-dir
RUN poetry config virtualenvs.in-project false
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-root --only main

COPY minecraft_copilot_ml/ minecraft_copilot_ml/
RUN poetry install --only main

ENTRYPOINT ["python", "minecraft_copilot_ml"]
