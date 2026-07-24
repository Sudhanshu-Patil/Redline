from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    anthropic_vision_model: str = "claude-sonnet-5"
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # Embeddings / retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    # Delta alignment thresholds
    alignment_exact_match_confidence: float = 0.95
    alignment_embedding_similarity_threshold: float = 0.75
    alignment_bbox_proximity_tolerance: float = 0.02  # fraction of page dimension

    # OCR
    ocr_confidence_threshold: float = 60.0  # tesseract confidence is 0-100
    ocr_dpi: int = 300

    # Paths
    data_dir: Path = Path("./data")
    traces_dir: Path = Path("./traces")
    chroma_persist_dir: Path = Path("./chroma_db")

    # Observability
    log_level: str = "INFO"


settings = Settings()
