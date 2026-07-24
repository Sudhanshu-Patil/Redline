.PHONY: sync lint typecheck test check run data chat eval eval-diff metrics dashboard

sync:
	uv sync

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint typecheck test

# --- Filled in as later phases land ---

run:
	uv run python -m src.ingest.pdf_native "data/samples/originals/lift_gas_compressor_26-KA-901.pdf" lift_gas_A

data:
	uv run python scripts/synthesize_pairs.py

chat:
	uv run python -m src.chat.answer $(MANIFEST)

eval:
	@echo "TODO: wire up in Phase 10 (make eval -> eval/scorecard.json)"

eval-diff:
	@echo "TODO: wire up in Phase 10 (compare two scorecard.json runs)"

metrics:
	uv run python -m src.observability.metrics_server

dashboard:
	@echo "TODO: wire up in Phase 11 (uvicorn src.dashboard.app:app)"
