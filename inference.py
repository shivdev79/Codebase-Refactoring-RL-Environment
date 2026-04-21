#!/usr/bin/env python3
"""
Inference Script — Codebase Refactoring RL Environment.

This script connects to a running environment server and drives an LLM
(OpenAI-compatible API, including HuggingFace TGI / vLLM endpoints) to
iteratively improve the buggy Python codebase over multiple steps.

How it works
------------
1.  Connect to the environment and call reset() to get the initial observation.
2.  Build a system prompt explaining the environment rules and reward signal.
3.  At each step, send the current codebase + test results to the LLM.
4.  Parse the LLM's JSON response into a CodeRefactorAction.
5.  Submit the action to the environment via step().
6.  Log the reward breakdown and test results.
7.  Repeat until all tests pass or max steps is reached.

Usage
-----
    # Requires the server to be running on port 8000
    python inference.py --base_url http://localhost:8000 --max_steps 20

    # Use a HuggingFace-compatible router instead of OpenAI
    python inference.py --api_base https://api-inference.huggingface.co/v1 \\
                        --model meta-llama/Llama-3-70B-Instruct \\
                        --max_steps 30

Environment variables
---------------------
    OPENAI_API_KEY   — API key for OpenAI (or HF token for HF router)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional import: openai (pip install openai)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print(
        "[inference.py] 'openai' package not found. "
        "Install with: pip install openai"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import environment client + models
# We support both in-package and standalone execution.
# ---------------------------------------------------------------------------
try:
    from rlproj.client import CodeRefactorEnv
    from rlproj.models import ActionType, CodeRefactorAction
except ImportError:
    # Running from the project root with PYTHONPATH set
    sys.path.insert(0, os.path.dirname(__file__))
    from client import CodeRefactorEnv          # type: ignore
    from models import ActionType, CodeRefactorAction  # type: ignore


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Python software engineer participating in a Reinforcement
Learning environment. Your task is to iteratively fix bugs and improve a
Python codebase so that all unit tests pass.

## Rules
- You may ONLY modify source files (not test files).
- You must submit exactly ONE action per response in valid JSON.
- Never introduce new bugs.

## Available action types
- "edit_function"   — Replace an entire function by name.
- "add_code"        — Append new code to a file.
- "replace_section" — Replace an exact verbatim block of code.
- "delete_section"  — Remove an exact verbatim block of code.

## Response format (strict JSON, no markdown fences)
{
  "action_type":    "<one of the action types above>",
  "target_file":    "<filename, e.g. calculator.py>",
  "function_name":  "<function name, required for edit_function, else null>",
  "new_code":       "<the new code string, or null>",
  "old_code":       "<exact existing code to find, or null>",
  "rationale":      "<brief explanation of this change>"
}

## Reward signal (for your information)
  +10  All tests pass (bonus, once per episode)
  +5   Per test you fix
  -10  Per test you break
  +3   You speed up execution
  +2   You improve lint score
  -5   Syntax/runtime error

Focus on fixing failing tests first. Think step by step."""


def build_user_prompt(obs) -> str:
    """
    Build the per-step user message from the current observation.

    Includes: summary, test results, and the current source files.
    """
    lines: List[str] = []

    lines.append(f"## Step Summary\n{obs.step_summary}\n")
    lines.append(
        f"## Test Results  ({obs.tests_passed}/{obs.tests_total} passing)\n"
    )

    for tr in obs.test_results:
        status = "✅ PASS" if tr.passed else "❌ FAIL"
        lines.append(f"  {status}  {tr.test_name}")
        if tr.error_message:
            # Trim long tracebacks to keep the prompt manageable
            msg = tr.error_message.strip()
            if len(msg) > 600:
                msg = msg[:600] + "\n... [truncated]"
            lines.append(f"         Error: {msg}")

    lines.append(f"\n## Quality Metrics")
    lines.append(f"  Lint score : {obs.lint_score:.1f}/10")
    lines.append(f"  Avg complexity : {obs.complexity_score:.1f}")
    lines.append(f"  Exec time : {obs.execution_time_seconds:.2f}s\n")

    lines.append("## Current Source Files\n")
    for fname, code in obs.codebase.items():
        lines.append(f"### {fname}\n```python\n{code}\n```\n")

    if obs.error_message:
        lines.append(f"\n⚠️ Previous action error: {obs.error_message}")

    lines.append("\nNow output your next action as JSON.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Interaction
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    """Call the LLM and return the raw text response."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def parse_action_from_llm(raw: str) -> Optional[CodeRefactorAction]:
    """
    Extract and validate a CodeRefactorAction from the LLM's raw text output.

    Strips markdown code fences if present, then parses JSON.
    Returns None if parsing fails.
    """
    # Strip markdown code fences (```json ... ```)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e}\nRaw output was:\n{raw[:500]}")
        return None

    # Map string action_type to the enum
    try:
        action_type = ActionType(data.get("action_type", ""))
    except ValueError:
        log.warning(f"Unknown action_type: {data.get('action_type')}")
        return None

    return CodeRefactorAction(
        action_type=action_type,
        target_file=data.get("target_file", ""),
        function_name=data.get("function_name"),
        new_code=data.get("new_code"),
        old_code=data.get("old_code"),
        rationale=data.get("rationale", ""),
    )


# ---------------------------------------------------------------------------
# Episode logging
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLog:
    """Accumulates per-step statistics for the full episode."""
    total_reward: float = 0.0
    steps: int = 0
    rewards: List[float] = field(default_factory=list)
    tests_passing_history: List[int] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    reward_breakdowns: List[Dict[str, float]] = field(default_factory=list)

    def record_step(self, reward: float, obs) -> None:
        self.steps += 1
        self.total_reward += reward
        self.rewards.append(round(reward, 2))
        self.tests_passing_history.append(obs.tests_passed)
        self.reward_breakdowns.append(obs.reward_breakdown)

    def print_summary(self) -> None:
        log.info("=" * 60)
        log.info("EPISODE SUMMARY")
        log.info("=" * 60)
        log.info(f"  Steps taken      : {self.steps}")
        log.info(f"  Total reward     : {self.total_reward:+.2f}")
        log.info(f"  Per-step rewards : {self.rewards}")
        log.info(f"  Tests passing    : {self.tests_passing_history}")
        log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_episode(
    env_url: str,
    llm_client: OpenAI,
    model: str,
    max_steps: int,
    temperature: float,
) -> EpisodeLog:
    """
    Run one full episode of the Codebase Refactoring RL Environment.

    Returns an EpisodeLog with all statistics.
    """
    episode_log = EpisodeLog()

    log.info(f"Connecting to environment at {env_url} ...")
    with CodeRefactorEnv(base_url=env_url) as env:

        # -- Reset --
        reset_result = env.reset()
        obs = reset_result.observation
        log.info(f"Episode started. {obs.step_summary}")
        log.info(
            f"  Baseline: {obs.tests_passed}/{obs.tests_total} tests passing."
        )

        # Conversation history for the LLM (maintains context across steps)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        for step in range(1, max_steps + 1):
            log.info(f"\n--- Step {step}/{max_steps} ---")

            # Build user message from current observation
            user_msg = build_user_prompt(obs)
            messages.append({"role": "user", "content": user_msg})

            # Query the LLM
            log.info("Querying LLM...")
            try:
                raw_response = call_llm(llm_client, model, messages, temperature)
            except Exception as e:
                log.error(f"LLM call failed: {e}")
                break

            log.info(f"LLM response (first 300 chars):\n{raw_response[:300]}")

            # Keep LLM response in conversation history
            messages.append({"role": "assistant", "content": raw_response})

            # Parse action
            action = parse_action_from_llm(raw_response)
            if action is None:
                log.warning("Could not parse action from LLM — skipping step.")
                continue

            log.info(
                f"Action: {action.action_type} on '{action.target_file}'"
                + (f" / fn='{action.function_name}'" if action.function_name else "")
            )
            log.info(f"Rationale: {action.rationale}")

            # Submit action to environment
            try:
                step_result = env.step(action)
            except Exception as e:
                log.error(f"env.step() failed: {e}")
                break

            obs = step_result.observation
            reward = step_result.reward or 0.0
            episode_log.record_step(reward, obs)
            episode_log.actions_taken.append(
                f"{action.action_type}:{action.target_file}"
            )

            # Log reward breakdown
            log.info(
                f"  Reward: {reward:+.2f}  |  "
                f"Tests: {obs.tests_passed}/{obs.tests_total}  |  "
                f"Lint: {obs.lint_score:.1f}"
            )
            if obs.reward_breakdown:
                log.info(f"  Breakdown: {obs.reward_breakdown}")
            if obs.error_message:
                log.warning(f"  Error: {obs.error_message}")

            # Check for episode completion
            if step_result.done:
                if obs.tests_failed == 0:
                    log.info("🎉 All tests pass! Episode complete.")
                else:
                    log.info("⏱️  Max steps reached. Episode ended.")
                break

        episode_log.print_summary()
        return episode_log


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference script for the Codebase Refactoring RL Environment."
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000",
        help="URL of the running environment server.",
    )
    parser.add_argument(
        "--api_base",
        default=None,
        help=(
            "Base URL for the OpenAI-compatible API. "
            "Defaults to the official OpenAI endpoint. "
            "Set to a HuggingFace TGI/vLLM URL for open-source models."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model name (e.g. gpt-4o, meta-llama/Llama-3-70B-Instruct).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Maximum number of environment steps per episode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run.",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    # Build OpenAI client (works with HuggingFace router or vLLM too)
    llm_kwargs: Dict[str, Any] = {"api_key": api_key}
    if args.api_base:
        llm_kwargs["base_url"] = args.api_base

    llm_client = OpenAI(**llm_kwargs)

    log.info(f"Model    : {args.model}")
    log.info(f"Env URL  : {args.base_url}")
    log.info(f"Episodes : {args.episodes}")
    log.info(f"Max steps: {args.max_steps}")

    all_logs: List[EpisodeLog] = []
    for ep in range(1, args.episodes + 1):
        log.info(f"\n{'='*60}")
        log.info(f"EPISODE {ep}/{args.episodes}")
        log.info(f"{'='*60}")
        ep_log = run_episode(
            env_url=args.base_url,
            llm_client=llm_client,
            model=args.model,
            max_steps=args.max_steps,
            temperature=args.temperature,
        )
        all_logs.append(ep_log)
        time.sleep(1)  # Brief pause between episodes

    # Final summary across all episodes
    if args.episodes > 1:
        avg_reward = sum(l.total_reward for l in all_logs) / len(all_logs)
        avg_steps = sum(l.steps for l in all_logs) / len(all_logs)
        log.info(f"\n{'='*60}")
        log.info(f"MULTI-EPISODE SUMMARY")
        log.info(f"  Episodes       : {args.episodes}")
        log.info(f"  Avg reward     : {avg_reward:+.2f}")
        log.info(f"  Avg steps      : {avg_steps:.1f}")
        log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
