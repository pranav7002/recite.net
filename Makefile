.PHONY: setup db db-down schema reset-db psql models index serve cli test lint quota \
        eval-tune eval-report eval-heldout eval-safety eval-router eval-rlm eval-all \
        latency-record latency-run latency-summarize

setup:          ## Create the venv and install dependencies
	python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

db:             ## Start Postgres + pgvector (localhost:5433)
	docker compose up -d db

db-down:        ## Stop Postgres (keeps the data volume)
	docker compose down

schema:         ## Apply backend/schema.sql to $$DATABASE_URL
	psql "$$DATABASE_URL" -f backend/schema.sql

reset-db:       ## Wipe the database, including embeddings (asks nothing — costs quota to rebuild)
	docker compose down -v && docker compose up -d db && sleep 3 && $(MAKE) schema

psql:           ## Open a psql shell on $$DATABASE_URL
	psql "$$DATABASE_URL"

models:         ## Download the local Whisper + Piper speech models into data/models/
	python scripts/download_models.py

index:          ## Index every PDF/.txt/.md in data/pdfs/ (no-op for files already indexed)
	python -m backend.ingest data/pdfs

serve:          ## Run the API + UI at http://localhost:8000, reloading on code changes
	uvicorn backend.api:app --reload --port 8000

cli:            ## Ask questions from the terminal (text only, no server needed)
	python -m backend.cli

test:           ## Lint, then run the unit tests
	ruff check . && pytest -q

lint:           ## Lint only
	ruff check .

quota:          ## Today's per-model call counts against their daily limits
	psql "$$DATABASE_URL" -c "SELECT model, calls FROM quota_daily WHERE day = CURRENT_DATE ORDER BY calls DESC;"

eval-tune:      ## Retrieval eval, embeddings arm, tune split (for chunk/prompt tuning)
	python -m evals.run_evals --suite retrieval --arm embeddings --split tune --runs 3

eval-report:    ## Retrieval eval, embeddings arm, report split (never used for tuning)
	python -m evals.run_evals --suite retrieval --arm embeddings --split report --runs 3

eval-heldout:   ## Retrieval eval, embeddings arm, held-out split (touch once, at the end)
	python -m evals.run_evals --suite retrieval --arm embeddings --split heldout --runs 3

eval-safety:    ## Safety suite (injection, legitimate-send, should-not-refuse cases)
	python -m evals.run_evals --suite safety --arm embeddings --runs 3

eval-router:    ## Retrieval eval through the router (escalates to RLM on weak matches) — not on the live path, see EXPERIMENTS.md
	python -m evals.run_evals --suite retrieval --arm router --split tune --runs 1

eval-rlm:       ## Retrieval eval, RLM arm only — slow (minutes/question) and not on the live path, see EXPERIMENTS.md
	python -m evals.run_evals --suite retrieval --arm rlm --split tune --runs 1

eval-all:       ## Full nightly sweep: both suites, embeddings arm, resumable
	python -m evals.run_evals --suite retrieval --arm embeddings --split report --runs 3 --resume
	python -m evals.run_evals --suite safety --arm embeddings --runs 3 --resume

latency-record: ## Speak the latency eval's questions to evals/audio/*.wav (macOS `say`, once)
	python -m evals.latency record

latency-run:    ## Replay them through /voice under every config — needs `make serve` running
	python -m evals.latency run --runs 5

latency-summarize: ## p50/p95 table -> evals/results/<date>_latency.md
	python -m evals.latency summarize
