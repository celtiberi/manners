# `probe`

Runtime checks and call-ordered debug logs for Python functions and LLM outputs. See [`SPEC.md`](./SPEC.md) for the full design.

## Install

```sh
uv tool install git+https://github.com/celtiberi/probe.git
```

In a project virtual environment:

```sh
uv add git+https://github.com/celtiberi/probe.git
```

## Quick Start

1. Attach checks to a function or intermediate step:

```python
from probe import probe

@probe(
    [
        "Tutor opens with a short Spanish greeting or phrase",
        "Tutor gives brief teaching or instructions to help the student",
        "Tutor asks the student to say something in Spanish",
        "Tutor does NOT give away the exact Spanish sentence the student is expected to say",
    ],
    id="tutor.opening_turn",
)
def generate_opening_turn(lesson, history) -> str:
    ...
```

Mid-function watchpoint:

```python
probe.here(
    [
        "Draft answer stays strictly grounded in retrieved chunks",
        "Draft answer directly addresses the user question",
    ],
    id="rag.draft_answer",
    kind=probe.type.text,
    when="draft cites specific figures or dates absent from retrieved_chunks",
    user_question=question,
    retrieved_chunks=chunks,
    draft=draft_text,
)
```

2. Run the pre-run check (populates `.probes/lock.json`):

```sh
probe check
```

3. Run your server or tests with probes enabled (either in the shell or by setting `PROBES=1` in `.env`):

```sh
PROBES=1 uvicorn app:app --reload
# or
PROBES=1 pytest
```

Each probe checks inputs/outputs inline. On failure (or `when=` match), `probe` opens `pdb` on that caller frame in an interactive terminal or raises `ProbeHaltError`.

4. Inspect results in call-entry order (works while a server is running too, via SQLite WAL):

```sh
probe report --all
probe log tutor.opening_turn rag.*
probe runs
```

## Configuration (`~/.config/probe/config.toml`)

Project overrides can also be placed under `[tool.probe]` in `pyproject.toml`.

```toml
[models.decision]
provider = "typesafe_jev"
url = "https://api.typesafe.ai/v1/systemone"
model = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"

[models.text]
provider = "openai_compat"
url = "https://api.deepseek.com/v1"
model = "deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"

# Local profile example:
# [models.decision]
# provider = "http_decision"
# url = "http://<host>.<tailnet>.ts.net:8780/v1/evaluate"
# model = "local-decision"
#
# [models.text]
# provider = "openai_compat"
# url = "http://<host>.<tailnet>.ts.net:11434/v1"
# model = "qwen3.5:9b"

[decision]
pass_threshold = 0.50

[text]
detail = "standard"
max_tokens = 400
explain_decision_failures = true
system_prompt = ""

[rate_limit]
max_calls = 5
window_seconds = 10.0
max_calls_per_run = -1

[check]
max_checks = 5
max_check_chars = 200
require_check = true

[redact]
env_files = [".env"]
extra_keys = []

[runtime]
db_path = ".probes/probes.db"
lock_path = ".probes/lock.json"
```

## License

Functional Source License, Version 1.1, with an Apache 2.0 future license (`FSL-1.1-Apache-2.0`). See [`LICENSE`](./LICENSE).
