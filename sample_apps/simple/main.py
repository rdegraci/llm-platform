"""Minimal sample: ask for a math expression and send it to the LLM.

Run from this directory (sample_apps/simple):

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...   # or LITELLM_API_KEY / .env
    python main.py

``requirements.txt`` installs llm-platform editable from the repo root (``-e ../..``).
``load_config()`` reads ``./llm_config.yaml`` in this directory.
"""
from pathlib import Path

from llm_platform import build_runtime, configure_litellm_api_key, load_config

# Resolve config relative to this file so the app works even if cwd differs.
_CONFIG_PATH = Path(__file__).resolve().parent / "llm_config.yaml"


def main() -> None:
    configure_litellm_api_key()
    runtime = build_runtime(load_config(str(_CONFIG_PATH)))

    expr = input("Math expression: ").strip()
    if not expr:
        raise SystemExit("No expression given.")

    answer = runtime.query(
        f"Compute this math expression and give only the result.\n\n{expr}"
    )
    print(answer)


if __name__ == "__main__":
    main()