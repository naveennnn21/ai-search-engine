from functools import lru_cache
from pathlib import Path

from pydantic import Field, PositiveFloat, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Search Engine API"
    upload_dir: Path = Field(default=Path("uploads"), validation_alias="UPLOAD_DIR")
    vector_index_dir: Path = Field(
        default=Path("storage/faiss"),
        validation_alias="VECTOR_INDEX_DIR",
    )
    faiss_index_filename: str = Field(
        default="documents.index",
        validation_alias="FAISS_INDEX_FILENAME",
    )
    vector_metadata_filename: str = Field(
        default="documents_metadata.json",
        validation_alias="VECTOR_METADATA_FILENAME",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        validation_alias="CORS_ORIGINS",
    )
    max_upload_size_mb: int = Field(default=25, validation_alias="MAX_UPLOAD_SIZE_MB")
    chunk_size: PositiveInt = Field(default=1000, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, ge=0, validation_alias="CHUNK_OVERLAP")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_dimensions: PositiveInt | None = Field(
        default=None,
        validation_alias="EMBEDDING_DIMENSIONS",
    )
    embedding_batch_size: PositiveInt = Field(default=64, validation_alias="EMBEDDING_BATCH_SIZE")
    embedding_max_retries: PositiveInt = Field(default=3, validation_alias="EMBEDDING_MAX_RETRIES")
    embedding_retry_base_seconds: PositiveFloat = Field(
        default=0.5,
        validation_alias="EMBEDDING_RETRY_BASE_SECONDS",
    )
    embedding_retry_max_seconds: PositiveFloat = Field(
        default=8.0,
        validation_alias="EMBEDDING_RETRY_MAX_SECONDS",
    )
    embedding_retry_jitter_seconds: float = Field(
        default=0.25,
        ge=0,
        validation_alias="EMBEDDING_RETRY_JITTER_SECONDS",
    )
    llm_model: str = Field(default="gpt-4.1-mini", validation_alias="LLM_MODEL")
    llm_max_retries: PositiveInt = Field(default=3, validation_alias="LLM_MAX_RETRIES")
    llm_retry_base_seconds: PositiveFloat = Field(
        default=0.5,
        validation_alias="LLM_RETRY_BASE_SECONDS",
    )
    llm_retry_max_seconds: PositiveFloat = Field(
        default=8.0,
        validation_alias="LLM_RETRY_MAX_SECONDS",
    )
    retrieval_candidate_count: PositiveInt = Field(
        default=40,
        validation_alias="RETRIEVAL_CANDIDATE_COUNT",
    )
    rerank_candidate_count: PositiveInt = Field(default=20, validation_alias="RERANK_CANDIDATE_COUNT")
    vector_search_weight: PositiveFloat = Field(default=0.65, validation_alias="VECTOR_SEARCH_WEIGHT")
    bm25_search_weight: PositiveFloat = Field(default=0.35, validation_alias="BM25_SEARCH_WEIGHT")
    enable_query_expansion: bool = Field(default=True, validation_alias="ENABLE_QUERY_EXPANSION")
    query_expansion_count: int = Field(default=2, ge=0, le=5, validation_alias="QUERY_EXPANSION_COUNT")
    enable_reranking: bool = Field(default=True, validation_alias="ENABLE_RERANKING")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_index_dir.mkdir(parents=True, exist_ok=True)
    return settings
