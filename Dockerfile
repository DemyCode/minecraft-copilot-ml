FROM python:3.10

WORKDIR /minecraft-copilot-ml

RUN pip install poetry --no-cache-dir && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    poetry run pip install -r requirements.txt --no-cache-dir

COPY minecraft_copilot_ml minecraft_copilot_ml
COPY README.md .
RUN poetry run pip install .

ENTRYPOINT ["./.venv/bin/python", "minecraft_copilot_ml"]
