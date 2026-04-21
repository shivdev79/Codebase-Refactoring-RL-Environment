#!/usr/bin/env python3
"""
Evaluation Pipeline — Codebase Refactoring RL Environment.

Runs a fixed, deterministic evaluation of the environment without an LLM.
Instead it applies a hand-coded optimal solution, verifying:
  - The reward function fires correctly at each step
  - Test counts change as expected
  - The environment terminates when all tests pass

This script is also useful for smoke-testing after any change to the
environment or sandbox logic.

Usage:
    python evaluate.py
    python evaluate.py --base_url http://localhost:8000   # against live server
    python evaluate.py --local                             # pure Python, no server
"""

import argparse
import logging
import sys
import os
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# The "oracle" — the correct fixed versions of all buggy functions
# ---------------------------------------------------------------------------

ORACLE_PATCHES: List[Tuple[str, str, str]] = [
    # (target_file, function_name, correct_source)
    (
        "calculator.py",
        "divide",
        '''\
def divide(a: float, b: float) -> float:
    """Return a divided by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
''',
    ),
    (
        "calculator.py",
        "fibonacci",
        '''\
def fibonacci(n: int) -> int:
    """
    Return the nth Fibonacci number (0-indexed).
    Uses iterative approach for O(n) time and O(1) space.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
    ),
    (
        "calculator.py",
        "find_max",
        '''\
def find_max(numbers: list) -> float:
    """Return the largest number in the list."""
    if not numbers:
        raise ValueError("Cannot find max of empty list.")
    return max(numbers)
''',
    ),
    (
        "calculator.py",
        "is_palindrome",
        '''\
def is_palindrome(s: str) -> bool:
    """
    Return True if s reads the same forwards and backwards,
    ignoring case and spaces.
    """
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]
''',
    ),
    (
        "data_processor.py",
        "normalize",
        '''\
def normalize(values: list[float]) -> list[float]:
    """Return min-max normalised values scaled to [0, 1]."""
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        # All values identical — return zeros instead of dividing by zero
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]
''',
    ),
    (
        "data_processor.py",
        "count_words",
        '''\
def count_words(text: str) -> int:
    """Return the number of words in text, stripping punctuation first."""
    import re
    if not text.strip():
        return 0
    # Remove punctuation attached to words, then split
    cleaned = re.sub(r"[^\\w\\s]", " ", text)
    return len(cleaned.split())
''',
    ),
    (
        "data_processor.py",
        "flatten_list",
        '''\
def flatten_list(nested: list) -> list:
    """Recursively flatten a nested list of arbitrary depth."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))   # recurse
        else:
            result.append(item)
    return result
''',
    ),
]


# ---------------------------------------------------------------------------
# Local (in-process) evaluation — no server needed
# ---------------------------------------------------------------------------

def run_local_evaluation() -> bool:
    """
    Run the environment entirely in-process using direct imports.
    No server or Docker required. Useful for rapid CI checks.

    Returns True if all tests pass at the end.
    """
    log.info("=" * 60)
    log.info("LOCAL EVALUATION MODE (in-process)")
    log.info("=" * 60)

    # Add project root to path so bare imports work
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    try:
        # Try installed-package imports first (uv run / pip install)
        from rlproj.server.rlproj_environment import CodeRefactorEnvironment
        from rlproj.models import ActionType, CodeRefactorAction
    except ImportError:
        try:
            # Fallback: direct module imports for script-mode execution
            # We import the individual .py files to avoid relative-import issues
            import importlib.util

            def _load_module(name, filepath):
                spec = importlib.util.spec_from_file_location(name, filepath)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                return mod

            models_mod = _load_module("models", os.path.join(project_dir, "models.py"))
            ActionType = models_mod.ActionType
            CodeRefactorAction = models_mod.CodeRefactorAction

            # Pre-register models so sandbox.py can import it
            _load_module("server.sandbox", os.path.join(project_dir, "server", "sandbox.py"))
            _load_module("server.sample_codebase", os.path.join(project_dir, "server", "sample_codebase.py"))
            env_mod = _load_module(
                "server.rlproj_environment",
                os.path.join(project_dir, "server", "rlproj_environment.py"),
            )
            CodeRefactorEnvironment = env_mod.CodeRefactorEnvironment
        except Exception as e:
            log.error(
                f"Could not import environment ({e}).\n"
                "Run from the 'rlproj' directory:\n"
                "  cd d:\\OpenEnvPrac\\rlagaent-\\rlproj\n"
                "  uv run --project . python evaluate.py --local"
            )
            return False

    env = CodeRefactorEnvironment()
    obs = env.reset()
    log.info(f"Reset complete. {obs.step_summary}")
    log.info(
        f"Baseline: {obs.tests_passed}/{obs.tests_total} passing, "
        f"lint={obs.lint_score:.1f}"
    )

    total_reward = 0.0
    all_passed = False

    for i, (target_file, func_name, new_src) in enumerate(ORACLE_PATCHES, start=1):
        action = CodeRefactorAction(
            action_type=ActionType.EDIT_FUNCTION,
            target_file=target_file,
            function_name=func_name,
            new_code=new_src,
            rationale=f"Oracle fix #{i}: correct implementation of {func_name}()",
        )

        obs = env.step(action)
        reward = obs.reward or 0.0
        total_reward += reward

        log.info(
            f"Step {i}: {func_name}() in {target_file} | "
            f"reward={reward:+.2f} | "
            f"tests={obs.tests_passed}/{obs.tests_total}"
        )
        if obs.reward_breakdown:
            log.info(f"         breakdown={obs.reward_breakdown}")
        if obs.error_message:
            log.warning(f"         ⚠ {obs.error_message}")

        if obs.done and obs.tests_failed == 0:
            all_passed = True
            break

    log.info("=" * 60)
    log.info(f"Total reward : {total_reward:+.2f}")
    log.info(f"All tests pass: {all_passed}")
    log.info("=" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# Remote (HTTP) evaluation — against a live server
# ---------------------------------------------------------------------------

def run_remote_evaluation(base_url: str) -> bool:
    """
    Run the oracle patch sequence against a live server via WebSocket.

    Returns True if all tests pass at the end.
    """
    log.info("=" * 60)
    log.info(f"REMOTE EVALUATION MODE — {base_url}")
    log.info("=" * 60)

    try:
        from client import CodeRefactorEnv
        from models import ActionType, CodeRefactorAction
    except ImportError:
        try:
            from rlproj.client import CodeRefactorEnv
            from rlproj.models import ActionType, CodeRefactorAction
        except ImportError:
            log.error("Cannot import client. Is rlproj installed?")
            return False

    total_reward = 0.0
    all_passed = False

    with CodeRefactorEnv(base_url=base_url) as env:
        reset_result = env.reset()
        obs = reset_result.observation
        log.info(f"Reset complete. {obs.step_summary}")

        for i, (target_file, func_name, new_src) in enumerate(ORACLE_PATCHES, start=1):
            action = CodeRefactorAction(
                action_type=ActionType.EDIT_FUNCTION,
                target_file=target_file,
                function_name=func_name,
                new_code=new_src,
                rationale=f"Oracle fix #{i}: correct implementation of {func_name}()",
            )

            step_result = env.step(action)
            obs = step_result.observation
            reward = step_result.reward or 0.0
            total_reward += reward

            log.info(
                f"Step {i}: {func_name}() | reward={reward:+.2f} | "
                f"tests={obs.tests_passed}/{obs.tests_total}"
            )
            if obs.reward_breakdown:
                log.info(f"  breakdown: {obs.reward_breakdown}")

            if step_result.done and obs.tests_failed == 0:
                all_passed = True
                break

    log.info("=" * 60)
    log.info(f"Total reward : {total_reward:+.2f}")
    log.info(f"All tests pass: {all_passed}")
    log.info("=" * 60)
    return all_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for the Codebase Refactoring RL Environment."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in-process without a server (faster, for CI).",
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000",
        help="URL of the running environment server (used when --local is not set).",
    )
    args = parser.parse_args()

    if args.local:
        success = run_local_evaluation()
    else:
        success = run_remote_evaluation(args.base_url)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
