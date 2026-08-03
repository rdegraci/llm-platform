"""Sample app with an app-local capability plugin (calculate).

Run from this directory (sample_apps/complex):

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...   # or LITELLM_API_KEY / .env
    python main.py

The ``calculate`` plugin lives under ``plugins/calculate/`` and is listed in
``llm_config.yaml``. The model can call the ``calculate`` tool instead of
doing the arithmetic itself.
"""
import sys
from pathlib import Path

from llm_platform import build_runtime, configure_litellm_api_key, load_config

_APP_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _APP_DIR / "llm_config.yaml"

# So ``plugins.calculate`` imports resolve when cwd is not this directory.
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


def main() -> None:
    configure_litellm_api_key()
    runtime = build_runtime(load_config(str(_CONFIG_PATH)))

    print("Tools:", sorted(runtime.tool_executor.tools))
    question = input("Ask a math question: ").strip()
    if not question:
        raise SystemExit("No question given.")

    answer = runtime.query(question)
    print(answer)


if __name__ == "__main__":
    main()
