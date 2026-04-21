"""
Data models for the Codebase Refactoring RL Environment.

The agent interacts with a Python codebase by sending structured actions
(edits, additions, deletions) and receiving rich observations including
test results, error messages, and performance metrics.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action types the agent can perform
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """All supported code-editing actions."""

    EDIT_FUNCTION = "edit_function"       # Replace the body of a specific function
    ADD_CODE = "add_code"                 # Append new code to a file
    REPLACE_SECTION = "replace_section"  # Replace a verbatim block of code
    DELETE_SECTION = "delete_section"    # Remove a verbatim block of code


class CodeRefactorAction(Action):
    """
    An action the agent submits to modify the codebase.

    The agent must specify:
    - Which type of action to perform
    - Which file to target (by name, e.g. 'calculator.py')
    - A human-readable rationale for the change
    - The relevant code snippets for the edit
    """

    # Type of modification
    action_type: ActionType = Field(
        ...,
        description="Which kind of edit to perform on the codebase.",
    )

    # Target file within the sandbox codebase
    target_file: str = Field(
        ...,
        description="Name of the file to modify (e.g. 'calculator.py').",
    )

    # For EDIT_FUNCTION: name of the function to replace entirely
    function_name: Optional[str] = Field(
        default=None,
        description="Name of the function to replace (required for EDIT_FUNCTION).",
    )

    # New code to inject (used by EDIT_FUNCTION, ADD_CODE, REPLACE_SECTION)
    new_code: Optional[str] = Field(
        default=None,
        description="The new code to insert or use as replacement.",
    )

    # Exact existing code to find (used by REPLACE_SECTION, DELETE_SECTION)
    old_code: Optional[str] = Field(
        default=None,
        description="The exact existing code block to find and operate on.",
    )

    # Brief explanation from the agent describing WHY this change is made
    rationale: str = Field(
        default="",
        description="Agent's explanation of why this change improves the codebase.",
    )


# ---------------------------------------------------------------------------
# Rich observation the environment sends back after each step
# ---------------------------------------------------------------------------


class TestResult(BaseModel):
    """Result for a single pytest test case."""

    test_name: str = Field(..., description="Full pytest node ID of the test.")
    passed: bool = Field(..., description="Whether the test passed.")
    error_message: Optional[str] = Field(
        default=None, description="Failure or error message if the test failed."
    )
    duration_seconds: float = Field(
        default=0.0, description="Time taken to run this test."
    )


class CodeRefactorObservation(Observation):
    """
    Observation returned by the environment after each step.

    Gives the agent full visibility into the current state of the codebase,
    what tests are passing, errors encountered, and quality metrics.
    """

    # The full current codebase as a mapping of {filename: source_code}
    codebase: Dict[str, str] = Field(
        default_factory=dict,
        description="Current state of all source files in the codebase.",
    )

    # Detailed per-test results from the latest pytest run
    test_results: List[TestResult] = Field(
        default_factory=list,
        description="Results of all unit tests after the latest action.",
    )

    # Aggregate statistics
    tests_passed: int = Field(default=0, description="Number of tests currently passing.")
    tests_failed: int = Field(default=0, description="Number of tests currently failing.")
    tests_total: int = Field(default=0, description="Total number of tests in the suite.")

    # Full pytest stdout + stderr for debugging
    test_output: str = Field(
        default="", description="Full captured output from the pytest run."
    )

    # Any syntax/runtime error produced when applying the action
    error_message: Optional[str] = Field(
        default=None, description="Syntax or runtime error from applying the action."
    )

    # Wall-clock time the test suite took to run
    execution_time_seconds: float = Field(
        default=0.0, description="Total wall-clock time for the test suite."
    )

    # Simple code-quality heuristics (flake8 lint score, complexity estimate)
    lint_score: float = Field(
        default=0.0,
        description="Lint quality score (0–10). Higher is better.",
    )
    complexity_score: float = Field(
        default=0.0,
        description="Average cyclomatic complexity (lower is better).",
    )

    # Human-readable summary of what happened this step
    step_summary: str = Field(
        default="", description="Human-readable summary of what changed this step."
    )

    # Reward breakdown for transparency / debugging
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Itemised reward components (e.g. 'test_pass_bonus': 5.0).",
    )
