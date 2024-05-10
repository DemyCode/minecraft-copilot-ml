FROM python:3.10

WORKDIR /minecraft-copilot-ml

RUN pip install poetry --no-cache-dir
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-cache-dir --no-root --only main

COPY minecraft_copilot_ml minecraft_copilot_ml
COPY README.md .
RUN poetry install --no-cache-dir --only main

ENTRYPOINT ["./.venv/bin/python", "minecraft_copilot_ml"]
