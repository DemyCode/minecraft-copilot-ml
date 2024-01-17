from pydantic import BaseSettings


class Settings(BaseSettings):
    X_HOME: str
    Y_HOME: str

    class Config:
        env_file = ".env"


settings = Settings()  # type: ignore
