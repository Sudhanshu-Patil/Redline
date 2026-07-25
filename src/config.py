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
    # Free tiers rate-limit aggressively with a stated wait; honor it up to
    # this many seconds per call before giving up (0 disables waiting).
    llm_rate_limit_max_wait_seconds: float = 120.0
    # $/MTok for cost telemetry (defaults = Sonnet 5 sticker; set 0 for free tiers).
    llm_cost_per_mtok_input: float = 3.0
    llm_cost_per_mtok_output: float = 15.0

    # Embeddings / retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    # Delta alignment thresholds. Calibrated against real Pair 1 edits (see
    # src/delta/align.py docstring): a same-slot value edit like "19057"->
    # "20500" scores only ~0.10 embedding similarity -- BELOW an unrelated
    # nearby element's ~0.25 -- so tight bbox proximity is treated as
    # decisive on its own, not merely a tiebreaker.
    alignment_embedding_similarity_threshold: float = 0.75
    alignment_bbox_proximity_tolerance: float = 0.02  # "tight": position alone is decisive
    alignment_tier3_loose_proximity: float = 0.15  # "loose": needs embedding similarity too
    alignment_min_embed_text_len: int = 2  # shorter text skips embedding, never matches in tier 3
    # Named block references (has a block_name) can move further and still be
    # "the same instance". Bare primitives (LINE/CIRCLE/... with no name) have
    # no identity beyond position, so they get a much tighter radius -- see
    # src/delta/align.py::geometry_match for the real case this fixes.
    geometry_match_max_bbox_distance: float = 0.3
    geometry_match_unnamed_max_bbox_distance: float = 0.03
    # Negative-control (Pair 4) detection threshold, against the exact-key
    # match rate specifically -- not overall alignment_rate, which two
    # unrelated documents sharing a P&ID template can inflate via
    # coincidental tier-3 position matches. Measured on real samples: 0.99
    # for a genuine revision pair, 0.24 for unrelated documents -- 0.5 sits
    # with comfortable margin on both sides.
    low_alignment_rate_threshold: float = 0.5
    low_alignment_min_elements: int = 20  # below this many *keyed* elements, skip the warning

    # Complementary warning, distinct from low_alignment_rate_threshold
    # above: that one fires on *mismatched* documents that both produce
    # plenty of keyed elements (tags/instrument-loops/valves/line-numbers)
    # but disagree; this one fires when a document produces almost none in
    # the first place -- e.g. a P&ID from a different drafting standard
    # whose tag-numbering convention the regex classifiers in
    # src/ingest/pdf_native.py don't recognize at all. Real Pair 1 measures
    # ~20% keyed; 5% is a conservative floor that only trips when
    # classification coverage is genuinely poor, not merely below-average.
    low_keyed_fraction_threshold: float = 0.05
    low_keyed_fraction_min_elements: int = 20  # below this many total elements, skip the warning

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
    dashboard_runs_dir: Path = Path("./dashboard_runs")

    # Observability
    log_level: str = "INFO"
    metrics_host: str = "127.0.0.1"
    metrics_port: int = 8000
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8001
    # Per-file cap on dashboard uploads. UploadFile.file.read() reads the
    # whole body into memory before writing it to disk, so an unbounded
    # upload is a real memory-exhaustion risk, not just a slow request.
    dashboard_max_upload_mb: int = 50


settings = Settings()
