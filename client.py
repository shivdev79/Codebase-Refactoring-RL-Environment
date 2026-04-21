"""
Codebase Refactoring RL Environment — Python Client.

Wraps the OpenEnv EnvClient to provide a typed, easy-to-use interface
for interacting with the server over WebSocket.

Usage:
    from rlproj.client import CodeRefactorEnv
    from rlproj.models import CodeRefactorAction, ActionType

    with CodeRefactorEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        print(result.observation.step_summary)

        action = CodeRefactorAction(
            action_type=ActionType.EDIT_FUNCTION,
            target_file="calculator.py",
            function_name="divide",
            new_code=(
                "def divide(a: float, b: float) -> float:\\n"
                "    if b == 0:\\n"
                "        raise ValueError('Cannot divide by zero.')\\n"
                "    return a / b\\n"
            ),
            rationale="Add zero-division guard to divide().",
        )
        result = env.step(action)
        print(f"Reward: {result.reward}")
        print(f"Tests passing: {result.observation.tests_passed}/{result.observation.tests_total}")
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CodeRefactorAction, CodeRefactorObservation, TestResult


class CodeRefactorEnv(EnvClient[CodeRefactorAction, CodeRefactorObservation, State]):
    """
    WebSocket client for the Codebase Refactoring RL Environment.

    Each instance maintains a persistent WebSocket connection to the server,
    enabling efficient multi-step interactions in a single episode.
    """

    # -----------------------------------------------------------------------
    # Action → JSON payload
    # -----------------------------------------------------------------------

    def _step_payload(self, action: CodeRefactorAction) -> Dict:
        """Convert a CodeRefactorAction to the JSON dict sent to the server."""
        return {
            "action_type": action.action_type,
            "target_file": action.target_file,
            "function_name": action.function_name,
            "new_code": action.new_code,
            "old_code": action.old_code,
            "rationale": action.rationale,
        }

    # -----------------------------------------------------------------------
    # JSON response → StepResult
    # -----------------------------------------------------------------------

    def _parse_result(self, payload: Dict) -> StepResult[CodeRefactorObservation]:
        """Parse the server's step-response JSON into a typed StepResult."""
        obs_data = payload.get("observation", {})

        # Reconstruct TestResult list
        test_results = [
            TestResult(
                test_name=t.get("test_name", "unknown"),
                passed=t.get("passed", False),
                error_message=t.get("error_message"),
                duration_seconds=t.get("duration_seconds", 0.0),
            )
            for t in obs_data.get("test_results", [])
        ]

        observation = CodeRefactorObservation(
            codebase=obs_data.get("codebase", {}),
            test_results=test_results,
            tests_passed=obs_data.get("tests_passed", 0),
            tests_failed=obs_data.get("tests_failed", 0),
            tests_total=obs_data.get("tests_total", 0),
            test_output=obs_data.get("test_output", ""),
            error_message=obs_data.get("error_message"),
            execution_time_seconds=obs_data.get("execution_time_seconds", 0.0),
            lint_score=obs_data.get("lint_score", 0.0),
            complexity_score=obs_data.get("complexity_score", 0.0),
            step_summary=obs_data.get("step_summary", ""),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    # -----------------------------------------------------------------------
    # JSON response → State
    # -----------------------------------------------------------------------

    def _parse_state(self, payload: Dict) -> State:
        """Parse the server's state-response JSON into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
