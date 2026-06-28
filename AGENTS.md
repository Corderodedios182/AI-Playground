# AI-Playground — AGENTS.md

Personal AI/ML learning playground. Multi-project Python repo. No CI/CD, no JS/Node.

## Repo structure

| Directory | What | How to run |
|-----------|------|------------|
| `analytics-playground/` | Data science template with tests | `pytest`, `black src/ tests/`, `ruff check src/ tests/`, `mypy src/` |
| `Azure-AI-Engineer-Associate/` | AI-102 study material | `pip install -r */requirements.txt` then `python *.py` (needs `.env`) |
| `Bootcamp+Completo+de+ChatGPT,LLM+y+LangChain/` | Spanish LLM bootcamp | `jupyter notebook` |
| `Cortex_AI_and_Streamlit/` | Snowflake Cortex + Streamlit | Run SQL scripts in Snowflake; app runs only inside Snowsight |
| `DeepLearning-AI/` | LangChain course notebooks | `jupyter notebook` |
| `fastapi-docker/` | Minimal FastAPI + Docker | `uvicorn app:app --reload` or `docker build/run` |
| `MCP_project/` | Bare MCP scaffold | `docker build -t mcp-project .` |
| `RAG_Demo_Architectures/` | Multiple RAG demos | Varies — see `rag-model-project/README.md` |
| `AWS/` | AWS Bedrock workshop | Deploy `cf.yml` via CloudFormation, run notebooks |

## Key conventions

- **Git identity** (repo-local): `Corderodedios182` / `corderodedios182@gmail.com`
- **Root `.gitignore`** ignores: venvs, `__pycache__`, `.env`, `.pem`, `.key`, `.sqlite3`, `.db`, `.csv`, logs, IDE folders, notebook checkpoints, pytest/coverage artifacts
- Most projects need API keys in `.env` files (OpenAI, Azure, HuggingFace, Snowflake, Ollama)
- Several projects are in **Spanish** (notebooks, comments, SQL instructions)
- No `package.json` anywhere — Python-only (except SQL/YAML configs)

## Testing/linting

Only `analytics-playground/` has tooling configured:
- `pytest` (discovers `tests/test_*.py`)
- `black src/ tests/`
- `ruff check src/ tests/`
- `mypy src/`

All other projects have no test/lint/typecheck setup.

## Gotchas

- `Cortex_AI_and_Streamlit` uses Snowflake-internal APIs (`_snowflake`, `get_active_session()`) — cannot run locally
- `RAG_Demo_Architectures/Talk_to_your_database...py` uses `eval()` on generated Pandas code (insecure with untrusted models)
- Several projects have placeholder code (`pass` bodies) waiting to be filled in
- `fastapi-docker` fetches Iris dataset live from GitHub Gist on every request (no caching)
