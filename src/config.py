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
    # Sticker $/MTok for the configured model (Sonnet 5); used for cost telemetry.
    llm_cost_per_mtok_input: float = 3.0
    llm_cost_per_mtok_output: float = 15.0

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
    # PSM 11 ("sparse text") measured best on line-art-dense P&IDs: finds all
    # ground-truth markers where the default PSM 3 layout analysis misses some.
    ocr_tesseract_psm: int = 11
    tesseract_cmd: str = ""  # explicit path to tesseract.exe; empty = rely on PATH

    # Vision-LLM fallback for low-confidence OCR regions
    vision_fallback_enabled: bool = True
    vision_fallback_max_regions: int = 12  # cap LLM calls per document
    vision_fallback_confidence: float = 0.9  # confidence assigned to LLM re-reads

    # Paths
    data_dir: Path = Path("./data")
    traces_dir: Path = Path("./traces")
    chroma_persist_dir: Path = Path("./chroma_db")

    # Observability
    log_level: str = "INFO"


settings = Settings()
