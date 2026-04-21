# Codebase Refactoring RL Environment

An OpenEnv-compatible reinforcement learning environment where an agent improves a realistic Python codebase by fixing bugs, preserving existing behavior, improving code quality, and maximizing unit-test success over multiple steps.

## What This Project Does

This environment simulates a practical software-engineering workflow:

- The agent receives a buggy multi-file Python codebase.
- A locked pytest suite evaluates behavior after every action.
- The environment returns structured observations with test results, execution time, errors, and quality metrics.
- The agent edits code iteratively until all tests pass or the episode ends.

This is meant for RL, agent-evaluation, and LLM-based code-improvement experiments rather than a toy one-shot benchmark.

## Project Structure

```text
.
|-- Dockerfile
|-- README.md
|-- __init__.py
|-- client.py
|-- evaluate.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- uv.lock
`-- server/
    |-- __init__.py
    |-- app.py
    |-- requirements.txt
    |-- rlproj_environment.py
    |-- sample_codebase.py
    `-- sandbox.py
```

## Core Components

### `models.py`
Defines the typed OpenEnv action and observation models:

- `CodeRefactorAction`
- `CodeRefactorObservation`
- `TestResult`

### `server/rlproj_environment.py`
Implements the RL environment:

- `reset()`
- `step()`
- reward computation
- episode state tracking

### `server/sandbox.py`
Provides safe execution utilities:

- applies agent edits to the in-memory codebase
- validates syntax with `ast`
- runs pytest in a temp sandbox
- computes lint and complexity heuristics

### `server/sample_codebase.py`
Contains:

- the buggy source files the agent can modify
- the locked test suite the agent cannot modify

### `server/app.py`
Wraps the environment as a FastAPI/OpenEnv service.

### `evaluate.py`
Runs a deterministic evaluation using an oracle patch sequence.

### `inference.py`
Runs an LLM loop against the environment using OpenAI-compatible APIs, including OpenAI or Hugging Face-compatible routers.

## Environment Design

### State

The environment internally tracks:

- current editable codebase
- locked test files
- previous passing tests
- previous execution and quality metrics
- episode id and step count

### Action Space

Supported action types:

- `edit_function`
- `add_code`
- `replace_section`
- `delete_section`

Each action includes a target file and the code payload needed for that edit.

### Observation Space

Each step returns:

- current codebase snapshot
- per-test results
- aggregate pass/fail counts
- full test output
- execution time
- lint score
- complexity score
- error messages
- reward breakdown

## Reward Function

The reward is intentionally shaped to reflect software-engineering progress:

- `+10` when all tests pass
- `+5` for each previously failing test fixed
- `-10` for each previously passing test broken
- `+3` for meaningful execution-time improvement
- `+2` for code-quality improvement
- `-5` for syntax or runtime failures
- `-5` for attempts to edit locked test files

## Safety and Anti-Reward-Hacking Rules

- Test files are locked and cannot be edited by the agent.
- Code is executed in a temporary sandbox directory.
- Syntax is validated before code is accepted.
- Pytest failures that prevent test collection are treated as runtime failures.
- Each step returns the full current observation so agents cannot hide regressions.

## Included Sample Task

The bundled sample codebase includes realistic bugs in:

- `calculator.py`
- `data_processor.py`

The initial baseline currently starts at:

- `15/25` tests passing

The oracle evaluation completes at:

- `25/25` tests passing
- total reward `+60.0`

## How To Run

### 1. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
pip install "openenv-core[core]" pytest flake8 openai uvicorn
```

### 2. Run the local evaluation

This is the fastest way to verify the environment works:

```bash
python evaluate.py --local
```

Expected outcome:

- baseline around `15/25`
- final result `All tests pass: True`

### 3. Start the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Test the running server

In another terminal:

```bash
python evaluate.py --base_url http://localhost:8000
```

### 5. Open API docs

Once the server is running:

```text
http://localhost:8000/docs
```

## Example API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

## LLM Inference

Run the environment with an OpenAI-compatible model:

```bash
python inference.py --base_url http://localhost:8000 --model gpt-4o --max_steps 20
```

Using a Hugging Face-compatible router:

```bash
python inference.py \
  --base_url http://localhost:8000 \
  --api_base https://api-inference.huggingface.co/v1 \
  --model meta-llama/Llama-3-70B-Instruct \
  --max_steps 30
```

Set your key first:

```bash
export OPENAI_API_KEY=your_key_here
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

## Docker

Build:

```bash
docker build -t codebase-refactoring-rl-environment .
```

Run:

```bash
docker run -p 8000:8000 codebase-refactoring-rl-environment
```

## Verified Local Checks

The project has been verified locally with:

```bash
python evaluate.py --local
python -m compileall -q .
```

## Example Interaction

```text
Reset  -> 15/25 tests passing
Step 1 -> fix divide()        -> reward +5.0
Step 2 -> fix fibonacci()     -> reward +5.0
Step 3 -> fix find_max()      -> reward +10.0
Step 4 -> fix is_palindrome() -> reward +10.0
Step 5 -> fix normalize()     -> reward +5.0
Step 6 -> fix count_words()   -> reward +5.0
Step 7 -> fix flatten_list()  -> reward +20.0
Done   -> 25/25 tests passing
```

## Publishing Notes

This repository is structured to be friendly for:

- OpenEnv serving
- local experimentation
- Hugging Face / OpenAI-compatible inference loops
- Docker-based packaging

If you publish this repo, the recommended landing point is this project root as the repository root.
