FROM python:3.10

WORKDIR /app

RUN pip install poetry --no-cache-dir
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /app/
RUN poetry export --without-hashes -f requirements.txt -o requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY . /app/
COPY minecraft_copilot_ml/ /app/minecraft_copilot_ml/
RUN poetry install --only main --no-interaction --no-ansi

ENTRYPOINT ["poetry", "run", "python", "minecraft_copilot_ml"]
