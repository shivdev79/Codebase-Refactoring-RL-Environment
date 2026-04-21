"""
Codebase Refactoring RL Environment — Core Logic.

Implements the OpenEnv Environment interface with:
  - reset():  Loads the initial buggy codebase, runs baseline tests
  - step():   Applies one agent action, runs tests, computes reward
  - state:    Returns current episode metadata

Reward Function
---------------
  +10  All tests pass (completion bonus, awarded once per episode)
  +5   Per test fixed this step  (was failing, now passing)
  -10  Per test broken this step (was passing, now failing)
  +3   Execution time improved by > 0.5 s relative to previous step
  +2   Lint score improved by > 0.5 points relative to previous step
  -5   Syntax / runtime error when applying the action
"""

import copy
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CodeRefactorAction, CodeRefactorObservation, TestResult
    from .sample_codebase import SOURCE_FILES, TEST_FILES
    from .sandbox import (
        apply_action_to_codebase,
        run_tests_in_sandbox,
        _compute_lint_score,
        _compute_complexity,
    )
except ImportError:
    from models import CodeRefactorAction, CodeRefactorObservation, TestResult
    from server.sample_codebase import SOURCE_FILES, TEST_FILES
    from server.sandbox import (
        apply_action_to_codebase,
        run_tests_in_sandbox,
        _compute_lint_score,
        _compute_complexity,
    )


# Maximum steps before an episode is force-terminated
MAX_STEPS_PER_EPISODE = 50


class CodeRefactorEnvironment(Environment):
    """
    Codebase Refactoring Reinforcement Learning Environment.

    Each episode starts with a fresh copy of the buggy codebase.
    The agent iteratively submits code-editing actions. After each action,
    the environment:
      1. Applies the edit to the in-memory codebase.
      2. Executes the locked test suite in a sandboxed subprocess.
      3. Computes a structured reward signal.
      4. Returns a rich observation.

    The episode ends when:
      - All tests pass (success), or
      - MAX_STEPS_PER_EPISODE is reached (timeout).
    """

    # Allow multiple concurrent WebSocket sessions (each gets own instance).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def __init__(self):
        """Initialise episode counters. reset() must be called before step()."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Live codebase the agent edits during the episode
        self._current_codebase: dict[str, str] = {}

        # Locked tests — never mutated
        self._test_files: dict[str, str] = copy.deepcopy(TEST_FILES)

        # Track which tests passed on the PREVIOUS step (for delta rewards)
        self._prev_passing: set[str] = set()

        # Metrics from the previous step (for delta rewards)
        self._prev_exec_time: float = 0.0
        self._prev_lint_score: float = 0.0
        self._prev_complexity: float = 0.0

        # Whether the all-tests-pass bonus has already been awarded this episode
        self._completion_bonus_given: bool = False

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(self) -> CodeRefactorObservation:
        """
        Reset the environment to episode start.

        Copies the original buggy source files, runs the baseline test suite,
        and returns the initial observation so the agent knows where it stands.
        """
        # Fresh episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_codebase = copy.deepcopy(SOURCE_FILES)
        self._test_files = copy.deepcopy(TEST_FILES)
        self._prev_passing = set()
        self._completion_bonus_given = False

        # Run baseline tests so the agent gets a starting picture
        results, output, elapsed = run_tests_in_sandbox(
            self._current_codebase, self._test_files
        )

        passing = {r.test_name for r in results if r.passed}
        self._prev_passing = passing
        self._prev_exec_time = elapsed
        self._prev_lint_score = _compute_lint_score(self._current_codebase)
        self._prev_complexity = _compute_complexity(self._current_codebase)

        n_pass = len(passing)
        n_fail = sum(1 for r in results if not r.passed)

        return CodeRefactorObservation(
            codebase=copy.deepcopy(self._current_codebase),
            test_results=results,
            tests_passed=n_pass,
            tests_failed=n_fail,
            tests_total=len(results),
            test_output=output,
            error_message=None,
            execution_time_seconds=round(elapsed, 3),
            lint_score=self._prev_lint_score,
            complexity_score=self._prev_complexity,
            step_summary=(
                f"Episode reset. Baseline: {n_pass}/{len(results)} tests passing. "
                f"Lint: {self._prev_lint_score:.1f}/10. "
                f"Avg complexity: {self._prev_complexity:.1f}."
            ),
            reward_breakdown={},
            done=False,
            reward=0.0,
        )

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(self, action: CodeRefactorAction) -> CodeRefactorObservation:  # type: ignore[override]
        """
        Apply one agent action and return a new observation with reward.

        Workflow:
          1. Validate that the agent is not trying to edit a test file.
          2. Apply the action to the codebase.
          3. Run the test suite in the sandbox.
          4. Compute the reward.
          5. Check for episode termination.
        """
        self._state.step_count += 1
        reward_breakdown: dict[str, float] = {}

        # -- Guard: agent must not touch test files --
        if action.target_file in self._test_files:
            obs = self._make_observation(
                error_message=(
                    f"INVALID ACTION: Agents are not allowed to modify test file "
                    f"'{action.target_file}'. Choose a source file."
                ),
                results=None,
                output="",
                elapsed=self._prev_exec_time,
                reward=-5.0,
                reward_breakdown={"invalid_test_edit_penalty": -5.0},
                step_summary="Action rejected: attempted to edit a locked test file.",
            )
            return obs

        # -- Apply the code edit --
        new_codebase, apply_error = apply_action_to_codebase(
            codebase=self._current_codebase,
            action_type=action.action_type,
            target_file=action.target_file,
            new_code=action.new_code,
            old_code=action.old_code,
            function_name=action.function_name,
        )

        # -- If the action itself caused a syntax/apply error, penalise --
        if apply_error:
            obs = self._make_observation(
                error_message=apply_error,
                results=None,   # re-use previous test state
                output="",
                elapsed=self._prev_exec_time,
                reward=-5.0,
                reward_breakdown={"syntax_error_penalty": -5.0},
                step_summary=f"Action failed: {apply_error}",
            )
            return obs

        # -- Run the test suite in the sandbox --
        results, output, elapsed = run_tests_in_sandbox(new_codebase, self._test_files)

        # -- Compute reward --
        total_reward, reward_breakdown = self._compute_reward(
            results, elapsed, new_codebase
        )

        # -- Update internal state only if action succeeded --
        self._current_codebase = new_codebase
        now_passing = {r.test_name for r in results if r.passed}
        self._prev_passing = now_passing
        self._prev_exec_time = elapsed

        new_lint = _compute_lint_score(new_codebase)
        new_complexity = _compute_complexity(new_codebase)
        self._prev_lint_score = new_lint
        self._prev_complexity = new_complexity

        n_pass = len(now_passing)
        n_fail = sum(1 for r in results if not r.passed)
        done = (n_fail == 0) or (self._state.step_count >= MAX_STEPS_PER_EPISODE)

        step_summary = (
            f"Step {self._state.step_count}: Applied '{action.action_type}' to "
            f"'{action.target_file}'. "
            f"Tests: {n_pass}/{len(results)} passing. "
            f"Reward: {total_reward:+.1f}. "
            f"Lint: {new_lint:.1f}/10."
        )

        return CodeRefactorObservation(
            codebase=copy.deepcopy(new_codebase),
            test_results=results,
            tests_passed=n_pass,
            tests_failed=n_fail,
            tests_total=len(results),
            test_output=output,
            error_message=None,
            execution_time_seconds=round(elapsed, 3),
            lint_score=new_lint,
            complexity_score=new_complexity,
            step_summary=step_summary,
            reward_breakdown=reward_breakdown,
            done=done,
            reward=total_reward,
        )

    # -----------------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------------

    def _compute_reward(
        self,
        results: list[TestResult],
        elapsed: float,
        new_codebase: dict[str, str],
    ) -> tuple[float, dict[str, float]]:
        """
        Compute the structured reward signal for this step.

        Returns (total_reward, breakdown_dict).
        """
        breakdown: dict[str, float] = {}
        total = 0.0

        now_passing = {r.test_name for r in results if r.passed}
        now_failing = {r.test_name for r in results if not r.passed}

        # +5 per test fixed (was failing, now passing)
        newly_fixed = now_passing - self._prev_passing
        if newly_fixed:
            bonus = 5.0 * len(newly_fixed)
            breakdown["tests_fixed_bonus"] = bonus
            total += bonus

        # -10 per test broken (was passing, now failing)
        newly_broken = self._prev_passing - now_passing
        if newly_broken:
            penalty = -10.0 * len(newly_broken)
            breakdown["tests_broken_penalty"] = penalty
            total += penalty

        # +10 completion bonus if ALL tests now pass (given only once)
        if results and not now_failing and not self._completion_bonus_given:
            breakdown["all_tests_pass_bonus"] = 10.0
            total += 10.0
            self._completion_bonus_given = True

        # -5 when pytest could not collect or execute the suite at all.
        if any(r.test_name.startswith("runtime_error::") for r in results):
            breakdown["runtime_error_penalty"] = -5.0
            total -= 5.0

        # +3 if execution time improved by more than 0.5 s
        if (self._prev_exec_time - elapsed) > 0.5:
            breakdown["speed_improvement_bonus"] = 3.0
            total += 3.0

        # +2 if lint score improved by more than 0.5 points
        new_lint = _compute_lint_score(new_codebase)
        if (new_lint - self._prev_lint_score) > 0.5:
            breakdown["lint_improvement_bonus"] = 2.0
            total += 2.0

        return round(total, 2), breakdown

    # -----------------------------------------------------------------------
    # _make_observation helper (used for early-return cases)
    # -----------------------------------------------------------------------

    def _make_observation(
        self,
        error_message,
        results,
        output,
        elapsed,
        reward,
        reward_breakdown,
        step_summary,
    ) -> CodeRefactorObservation:
        """Build an observation for an action that did not mutate the codebase."""
        # Re-run tests on the current codebase so rejected/invalid actions still
        # return a complete, truthful observation.
        test_results, out, actual_elapsed = run_tests_in_sandbox(
            self._current_codebase, self._test_files
        )
        output = output or out
        elapsed = actual_elapsed if elapsed is None else elapsed

        n_pass = sum(1 for r in test_results if r.passed)
        n_fail = sum(1 for r in test_results if not r.passed)

        return CodeRefactorObservation(
            codebase=copy.deepcopy(self._current_codebase),
            test_results=test_results,
            tests_passed=n_pass,
            tests_failed=n_fail,
            tests_total=len(test_results),
            test_output=output,
            error_message=error_message,
            execution_time_seconds=round(elapsed, 3),
            lint_score=self._prev_lint_score,
            complexity_score=self._prev_complexity,
            step_summary=step_summary,
            reward_breakdown=reward_breakdown,
            done=False,
            reward=reward,
        )

    # -----------------------------------------------------------------------
    # state property (required by OpenEnv interface)
    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return current episode metadata."""
        return self._state
