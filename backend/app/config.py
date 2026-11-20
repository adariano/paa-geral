from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "historia_brasil"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    top_k_results: int = 4
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    mock_mode: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
