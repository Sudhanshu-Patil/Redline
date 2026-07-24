from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider selection. "anthropic" uses the anthropic SDK with
    # ANTHROPIC_* settings; "openai_compatible" works with any provider that
    # speaks the OpenAI chat-completions format (Groq, Moonshot/Kimi,
    # OpenRouter, ...) via LLM_BASE_URL/LLM_API_KEY/LLM_MODEL.
    llm_provider: Literal["anthropic", "openai_compatible"] = "anthropic"

    # -- anthropic backend --
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    anthropic_vision_model: str = "claude-sonnet-5"

    # -- openai_compatible backend --
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_vision_model: str = "qwen/qwen3.6-27b"
    # Reasoning models spend hidden tokens before answering; vision reads need
    # headroom well beyond the answer length (observed: hard P&ID crops
    # exceed 1500 thinking tokens and truncate).
    llm_vision_max_tokens: int = 3000
    # Send Groq's reasoning_format=hidden to suppress <think> output. Set
    # false for providers that reject the parameter (think-tags are stripped
    # from replies regardless).
    llm_vision_reasoning_hidden: bool = True

    # -- shared --
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0
    # $/MTok for cost telemetry (defaults = Sonnet 5 sticker; set 0 for free tiers).
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

    # DWG/DXF
    oda_file_converter: str = ""  # path to ODAFileConverter.exe; empty = PATH lookup
    dwg_dim_decimals: int = 1  # decimals when rendering measured dimension values

    # Paths
    data_dir: Path = Path("./data")
    traces_dir: Path = Path("./traces")
    chroma_persist_dir: Path = Path("./chroma_db")

    # Observability
    log_level: str = "INFO"


settings = Settings()
