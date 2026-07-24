.PHONY: sync lint typecheck test check run chat eval eval-diff metrics dashboard

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
	@echo "TODO: wire up in Phase 1 (ingest a document -> canonical JSON)"

chat:
	@echo "TODO: wire up in Phase 8 (grounded chat CLI)"

eval:
	@echo "TODO: wire up in Phase 10 (make eval -> eval/scorecard.json)"

eval-diff:
	@echo "TODO: wire up in Phase 10 (compare two scorecard.json runs)"

metrics:
	@echo "TODO: wire up in Phase 7 (serve /metrics)"

dashboard:
	@echo "TODO: wire up in Phase 11 (uvicorn src.dashboard.app:app)"
