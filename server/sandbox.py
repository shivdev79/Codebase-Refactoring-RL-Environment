"""
Sandbox Executor for the Codebase Refactoring RL Environment.

Responsibilities:
  1. Write the current codebase files to a temporary directory.
  2. Run pytest against a COPY of the test suite (agent cannot modify tests).
  3. Parse pytest JSON output into structured TestResult objects.
  4. Measure execution time and compute code-quality heuristics.
  5. Clean up the temp directory afterwards.

Security model:
  - The sandbox writes only to a controlled temp dir.
  - Test files are injected from a read-only reference — the agent is
    NEVER given write access to them.
  - Subprocess execution is time-limited to prevent infinite loops.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from ..models import TestResult
except ImportError:
    from models import TestResult


# Maximum wall-clock seconds a test run may take before being killed.
SANDBOX_TIMEOUT_SECONDS = 30


def _compute_lint_score(source_files: Dict[str, str]) -> float:
    """
    Compute a 0-10 lint quality score by running flake8 on the source.

    Higher score == fewer lint errors per 100 lines of code.
    Returns 10.0 if flake8 is not installed (fail-safe).
    """
    if not source_files:
        return 10.0

    # Write files to a tiny temp directory just for linting
    with tempfile.TemporaryDirectory(prefix="rl_lint_") as lint_dir:
        total_lines = 0
        for fname, code in source_files.items():
            fpath = Path(lint_dir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")
            total_lines += code.count("\n") + 1

        try:
            result = subprocess.run(
                [sys.executable, "-m", "flake8", "--max-line-length=120", lint_dir],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Count number of lint violations
            violations = len([l for l in result.stdout.splitlines() if l.strip()])
            # Normalise: score out of 10, capped at 0
            score = max(0.0, 10.0 - (violations / max(total_lines, 1)) * 100)
            return round(score, 2)
        except Exception:
            return 10.0  # flake8 unavailable — don't penalise


def _compute_complexity(source_files: Dict[str, str]) -> float:
    """
    Estimate average cyclomatic complexity using a simple AST walk.

    Counts decision-point keywords (if, for, while, except, and, or, with)
    across all functions and averages them.  Returns 0.0 if no functions found.
    """
    decision_keywords = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.BoolOp, ast.With, ast.AsyncFor, ast.AsyncWith,
    )
    total_complexity = 0
    function_count = 0

    for code in source_files.values():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                complexity = 1  # base complexity
                for child in ast.walk(node):
                    if isinstance(child, decision_keywords):
                        complexity += 1
                total_complexity += complexity

    if function_count == 0:
        return 0.0
    return round(total_complexity / function_count, 2)


def _parse_pytest_json(report_path: str) -> List[TestResult]:
    """
    Parse a pytest JSON report into a list of TestResult objects.

    Uses the pytest-json-report plugin output format.
    Falls back to empty list if the file cannot be read.
    """
    import json

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    results: List[TestResult] = []
    for test in data.get("tests", []):
        node_id = test.get("nodeid", "unknown")
        outcome = test.get("outcome", "failed")
        passed = outcome == "passed"
        duration = test.get("call", {}).get("duration", 0.0) if "call" in test else 0.0

        # Collect any longrepr (traceback text) from setup/call/teardown
        error_parts: List[str] = []
        for phase in ("setup", "call", "teardown"):
            phase_data = test.get(phase, {})
            if phase_data.get("longrepr"):
                error_parts.append(phase_data["longrepr"])

        results.append(
            TestResult(
                test_name=node_id,
                passed=passed,
                error_message="\n".join(error_parts) if error_parts else None,
                duration_seconds=round(duration, 4),
            )
        )
    return results


def run_tests_in_sandbox(
    source_files: Dict[str, str],
    test_files: Dict[str, str],
) -> Tuple[List[TestResult], str, float]:
    """
    Execute pytest inside a sandboxed temp directory.

    Args:
        source_files: Mapping of {filename: source_code} for the agent's codebase.
        test_files:   Mapping of {filename: source_code} for the LOCKED test suite.

    Returns:
        (test_results, captured_output, wall_clock_seconds)
    """
    with tempfile.TemporaryDirectory(prefix="rl_sandbox_") as sandbox_dir:
        # -- Write source files the agent controls --
        for fname, code in source_files.items():
            fpath = Path(sandbox_dir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")

        # -- Write LOCKED test files (agent cannot modify these) --
        for fname, code in test_files.items():
            fpath = Path(sandbox_dir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")

        # -- Add a minimal conftest so imports resolve correctly --
        conftest = Path(sandbox_dir) / "conftest.py"
        conftest.write_text(
            "import sys, os\n"
            "sys.path.insert(0, os.path.dirname(__file__))\n",
            encoding="utf-8",
        )

        report_path = os.path.join(sandbox_dir, "report.json")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            "--tb=short",
            "--disable-warnings",
            sandbox_dir,
        ]

        json_report_available = _pytest_json_report_available()
        if json_report_available:
            cmd.insert(4, f"--json-report-file={report_path}")
            cmd.insert(4, "--json-report")

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT_SECONDS,
                cwd=sandbox_dir,
            )
            elapsed = time.monotonic() - start
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            elapsed = SANDBOX_TIMEOUT_SECONDS
            output = f"[SANDBOX] Test run timed out after {SANDBOX_TIMEOUT_SECONDS}s."
            return [], output, elapsed

        results = _parse_pytest_json(report_path) if json_report_available else []

        # Fallback: parse verbose pytest output when json-report is unavailable.
        if not results:
            results = _parse_pytest_stdout(output)

        # Never allow "0 tests" to look like success. If pytest failed before
        # reporting individual tests, return one synthetic failed result.
        if not results and proc.returncode != 0:
            results = [
                TestResult(
                    test_name="runtime_error::pytest",
                    passed=False,
                    error_message=output[-4000:] if output else "pytest failed before reporting tests.",
                    duration_seconds=round(elapsed, 4),
                )
            ]

        return results, output, elapsed


def _pytest_json_report_available() -> bool:
    """Return True when pytest-json-report can be imported in this interpreter."""
    try:
        import pytest_jsonreport  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _parse_pytest_stdout(output: str) -> List[TestResult]:
    """
    Minimal fallback parser for plain pytest -q output.

    Handles lines like:
        PASSED test_calculator.py::test_add
        FAILED test_calculator.py::test_divide - AssertionError: ...
    """
    results: List[TestResult] = []
    pattern = re.compile(r"^(?P<node>\S+::\S+)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED)\b")
    for line in output.splitlines():
        line = line.strip()
        match = pattern.match(line)
        if match:
            status = match.group("status")
            if status == "SKIPPED":
                continue
            results.append(
                TestResult(
                    test_name=match.group("node"),
                    passed=status == "PASSED",
                    error_message=None if status == "PASSED" else _failure_excerpt(output, match.group("node")),
                    duration_seconds=0.0,
                )
            )
    return results


def _failure_excerpt(output: str, node_id: str) -> Optional[str]:
    """Extract a compact failure snippet for a pytest node from verbose output."""
    marker = f"FAILED {node_id}"
    idx = output.find(marker)
    if idx == -1:
        marker = f"ERROR {node_id}"
        idx = output.find(marker)
    if idx == -1:
        return None
    return output[idx:idx + 1200].strip()


def apply_action_to_codebase(
    codebase: Dict[str, str],
    action_type: str,
    target_file: str,
    new_code: Optional[str] = None,
    old_code: Optional[str] = None,
    function_name: Optional[str] = None,
) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Apply one agent action to a codebase dict (immutably — returns a new dict).

    Returns:
        (new_codebase, error_message)  where error_message is None on success.
    """
    import copy
    new_codebase = copy.deepcopy(codebase)

    action_value = getattr(action_type, "value", action_type)

    # Ensure file exists (for edits; ADD_CODE may create a new file)
    if action_value != "add_code" and target_file not in new_codebase:
        return codebase, f"Target file '{target_file}' not found in codebase."

    current_code = new_codebase.get(target_file, "")

    if action_value == "edit_function":
        if not function_name or not new_code:
            return codebase, "edit_function requires 'function_name' and 'new_code'."
        result, err = _replace_function(current_code, function_name, new_code)
        if err:
            return codebase, err
        new_codebase[target_file] = result

    elif action_value == "add_code":
        if not new_code:
            return codebase, "add_code requires 'new_code'."
        new_codebase[target_file] = current_code + "\n\n" + new_code

    elif action_value == "replace_section":
        if not old_code or not new_code:
            return codebase, "replace_section requires 'old_code' and 'new_code'."
        if old_code not in current_code:
            return codebase, f"Could not find the specified 'old_code' block in '{target_file}'."
        new_codebase[target_file] = current_code.replace(old_code, new_code, 1)

    elif action_value == "delete_section":
        if not old_code:
            return codebase, "delete_section requires 'old_code'."
        if old_code not in current_code:
            return codebase, f"Could not find the specified 'old_code' block in '{target_file}'."
        new_codebase[target_file] = current_code.replace(old_code, "", 1)

    else:
        return codebase, f"Unknown action_type: '{action_type}'."

    # Validate syntax of the modified file
    try:
        ast.parse(new_codebase[target_file])
    except SyntaxError as e:
        return codebase, f"SyntaxError after applying action: {e}"

    return new_codebase, None


def _replace_function(source: str, func_name: str, new_func_code: str) -> Tuple[str, Optional[str]]:
    """
    Replace the entire definition of `func_name` in `source` with `new_func_code`.

    Uses AST to locate the exact line range of the function, then does a
    line-based substitution.  Returns (new_source, error_string).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return source, f"Cannot parse source file: {e}"

    lines = source.splitlines(keepends=True)
    target_node = None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                target_node = node
                break

    if target_node is None:
        return source, f"Function '{func_name}' not found in the file."

    # ast line numbers are 1-indexed; end_lineno requires Python 3.8+
    start = target_node.lineno - 1
    end = getattr(target_node, "end_lineno", None)
    if end is None:
        return source, "Python 3.8+ required for end_lineno support."

    # Preserve indentation prefix of the original function definition
    original_indent = re.match(r"(\s*)", lines[start]).group(1)
    new_lines = []
    for i, line in enumerate(new_func_code.splitlines(keepends=True)):
        if i == 0:
            new_lines.append(original_indent + line.lstrip())
        else:
            new_lines.append(line)

    # Ensure trailing newline
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    replaced = lines[:start] + new_lines + lines[end:]
    return "".join(replaced), None
