from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Search Engine API"
    upload_dir: Path = Field(default=Path("uploads"), validation_alias="UPLOAD_DIR")
    max_upload_size_mb: int = Field(default=25, validation_alias="MAX_UPLOAD_SIZE_MB")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
