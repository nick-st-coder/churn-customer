# AGENTS.md

## Project purpose

This repository trains and serves a binary customer-churn prediction model. The business-facing API lives in [app/main.py](app/main.py), while the ML pipeline and reusable code live under [src](src).

Use the repository docs in [README.md](README.md) for background and product context. Use [pyproject.toml](pyproject.toml) for the declared Python dependencies and test tooling.

## Working conventions

- Keep code organized around the pipeline layers already used by the repo:
  - data ingestion/preprocessing under [src/data](src/data)
  - feature construction under [src/features](src/features)
  - training/tuning under [src/models](src/models)
  - inference/serving under [src/serving](src/serving)
  - shared validation/helpers under [src/utils](src/utils)
- Prefer extending the existing module boundaries instead of introducing new top-level packages unless the task clearly requires it.
- Treat the notebooks as exploratory artifacts and not the main source of truth for production logic.

## Local development and verification

- Install project dependencies with the package metadata in [pyproject.toml](pyproject.toml).
- Run the Python test suite with:

  ```bash
  python -m pytest
  ```

- Run the API locally with:

  ```bash
  uvicorn app.main:app --reload
  ```

- For model training/tuning work, expect MLflow usage in [src/models/train.py](src/models/train.py) and [src/models/tuning.py](src/models/tuning.py). The default tracking URI is `http://127.0.0.1:5000`.
- For inference work, [src/serving/inference.py](src/serving/inference.py) currently points to a registered MLflow model alias/name pattern and should be preserved unless the task explicitly changes serving behavior.

## Common pitfalls

- The project uses a FastAPI app in [app/main.py](app/main.py). Keep the API contract and request/response shape consistent with that file.
- The repository is Python-first; model code, utilities, and feature engineering are implemented under [src](src), not under ad-hoc scripts.
- When changing data contracts or feature engineering, check the corresponding pipeline inputs and outputs across [src/data](src/data), [src/features](src/features), and [src/models](src/models) together.

## Preferred change scope

- For bug fixes, make the minimum necessary change in the existing module that owns the behavior.
- For new features, add them in the logical layer that already owns the functionality, then wire them through the calling module rather than bypassing the pipeline.
- If a new file is needed, prefer a path that matches the existing project structure instead of introducing a parallel pattern.
