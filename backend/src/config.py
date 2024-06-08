from typing import Optional

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_ENDPOINT: str
    LANGCHAIN_API_KEY: str
    TAVILY_API_KEY: str

    # Database connection settings
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASS: SecretStr
    DB_NAME: str
    DB_URL: Optional[str] = None

    @field_validator("DB_URL", mode="after")
    @staticmethod
    def assemble_db_url(v: Optional[str], info: ValidationInfo) -> str:
        """Construct db host url."""
        if isinstance(v, str):
            return v

        return f"mongodb://{info.data.get('DB_USER')}:{info.data.get('DB_PASS').get_secret_value()}@{info.data.get('DB_HOST')}/{info.data.get('DB_NAME')}?authSource=admin"  # type: ignore


settings = Settings()  # type: ignore
