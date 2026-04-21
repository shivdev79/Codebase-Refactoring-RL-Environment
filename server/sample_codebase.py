"""
Sample codebase and test suite for the Codebase Refactoring RL Environment.

The SOURCE_FILES dict represents the starting state — a small Python module
with several deliberate bugs, performance issues, and style violations.

The TEST_FILES dict is the LOCKED test suite.  The agent is NEVER allowed
to modify these files.  All reward signal flows from whether the tests pass.

Bug inventory (for reference — the agent has to discover these itself):
  calculator.py
    - divide(): does not handle division by zero (raises ZeroDivisionError)
    - fibonacci(): uses inefficient recursive approach AND has an off-by-one
      error (returns fib(n-1) instead of fib(n))
    - find_max(): returns the minimum instead of maximum
    - is_palindrome(): fails for strings with mixed case and spaces

  data_processor.py
    - normalize(): divides by (max - min) but crashes when all values are equal
    - count_words(): splits on whitespace only — misses punctuation-attached words
    - flatten_list(): only flattens one level deep, not recursively
"""

# ---------------------------------------------------------------------------
# The agent-controlled codebase (starts with bugs)
# ---------------------------------------------------------------------------

SOURCE_FILES: dict[str, str] = {
    "calculator.py": '''\
"""
A utility module providing common mathematical operations.
Contains intentional bugs for the RL agent to discover and fix.
"""


def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return a minus b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of a and b."""
    return a * b


def divide(a: float, b: float) -> float:
    """
    Return a divided by b.
    BUG: Does not handle b == 0 — raises ZeroDivisionError instead of
    raising a descriptive ValueError.
    """
    return a / b  # BUG: no zero-check


def fibonacci(n: int) -> int:
    """
    Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, ...).
    BUG 1: Off-by-one — returns fib(n-1) instead of fib(n).
    BUG 2: Exponential time complexity; should use memoisation or iteration.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 3) + fibonacci(n - 2)  # BUG: off-by-one, slow


def find_max(numbers: list) -> float:
    """
    Return the largest number in the list.
    BUG: Uses min() instead of max().
    """
    if not numbers:
        raise ValueError("Cannot find max of empty list.")
    return min(numbers)  # BUG: should be max()


def is_palindrome(s: str) -> bool:
    """
    Return True if s reads the same forwards and backwards.
    BUG: Does not normalise case or strip spaces, so
    'Racecar' and 'race car' incorrectly return False.
    """
    return s == s[::-1]  # BUG: no .lower().replace(" ", "")
''',

    "data_processor.py": '''\
"""
Data processing utilities.
Contains intentional bugs for the RL agent to discover and fix.
"""

from typing import Any


def normalize(values: list[float]) -> list[float]:
    """
    Return min-max normalised values scaled to [0, 1].
    BUG: Crashes with ZeroDivisionError when all values are identical
    (max - min == 0).
    """
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]  # BUG: no zero-range guard


def count_words(text: str) -> int:
    """
    Return the number of words in text.
    BUG: Splits only on whitespace so 'hello,world' is counted as 1 word.
    A proper implementation should strip punctuation first.
    """
    if not text.strip():
        return 0
    return len(text.split())  # BUG: punctuation not stripped


def flatten_list(nested: list[Any]) -> list[Any]:
    """
    Recursively flatten a nested list of arbitrary depth.
    BUG: Only flattens one level — [[1,[2]],3] → [1,[2],3] instead of [1,2,3].
    """
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)   # BUG: should recurse, not just extend
        else:
            result.append(item)
    return result


def calculate_statistics(numbers: list[float]) -> dict:
    """
    Return basic descriptive statistics for a list of numbers.
    This function is correct — agent should leave it alone.
    """
    if not numbers:
        return {"mean": 0, "median": 0, "std": 0}
    n = len(numbers)
    mean = sum(numbers) / n
    sorted_nums = sorted(numbers)
    mid = n // 2
    median = sorted_nums[mid] if n % 2 != 0 else (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std = variance ** 0.5
    return {"mean": round(mean, 4), "median": median, "std": round(std, 4)}
''',
}

# ---------------------------------------------------------------------------
# LOCKED test suite — agent cannot modify these files
# ---------------------------------------------------------------------------

TEST_FILES: dict[str, str] = {
    "test_calculator.py": '''\
"""Unit tests for calculator.py — locked, agent cannot modify."""
import pytest
from calculator import add, subtract, multiply, divide, fibonacci, find_max, is_palindrome


# --- Basic arithmetic ---

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0.1, 0.2) == pytest.approx(0.3, rel=1e-6)


def test_subtract():
    assert subtract(10, 4) == 6
    assert subtract(0, 5) == -5


def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 5) == -10
    assert multiply(0, 999) == 0


# --- divide ---

def test_divide_normal():
    assert divide(10, 2) == 5.0
    assert divide(7, 2) == pytest.approx(3.5)


def test_divide_by_zero():
    """Agent must add a guard that raises ValueError (not ZeroDivisionError)."""
    with pytest.raises(ValueError):
        divide(1, 0)


# --- fibonacci ---

def test_fibonacci_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_fibonacci_sequence():
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    for i, val in enumerate(expected):
        assert fibonacci(i) == val, f"fibonacci({i}) should be {val}"


# --- find_max ---

def test_find_max_basic():
    assert find_max([3, 1, 4, 1, 5, 9, 2, 6]) == 9


def test_find_max_negatives():
    assert find_max([-5, -1, -3]) == -1


def test_find_max_single():
    assert find_max([42]) == 42


def test_find_max_empty():
    with pytest.raises(ValueError):
        find_max([])


# --- is_palindrome ---

def test_is_palindrome_simple():
    assert is_palindrome("racecar") is True
    assert is_palindrome("hello") is False


def test_is_palindrome_case_insensitive():
    assert is_palindrome("Racecar") is True


def test_is_palindrome_with_spaces():
    assert is_palindrome("race car") is True
    assert is_palindrome("A man a plan a canal Panama") is True
''',

    "test_data_processor.py": '''\
"""Unit tests for data_processor.py — locked, agent cannot modify."""
import pytest
from data_processor import normalize, count_words, flatten_list, calculate_statistics


# --- normalize ---

def test_normalize_basic():
    result = normalize([0.0, 5.0, 10.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_identical_values():
    """Agent must handle the zero-range edge case."""
    result = normalize([5.0, 5.0, 5.0])
    assert result == [0.0, 0.0, 0.0]


def test_normalize_empty():
    assert normalize([]) == []


# --- count_words ---

def test_count_words_simple():
    assert count_words("hello world") == 2


def test_count_words_with_punctuation():
    """Agent must strip punctuation so 'hello,world' counts as 2 words."""
    assert count_words("hello,world") == 2


def test_count_words_empty():
    assert count_words("   ") == 0


# --- flatten_list ---

def test_flatten_one_level():
    assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_flatten_deep():
    """Agent must implement true recursive flattening."""
    assert flatten_list([1, [2, [3, [4]]]]) == [1, 2, 3, 4]


def test_flatten_mixed():
    assert flatten_list([1, [2, 3], 4, [5, [6]]]) == [1, 2, 3, 4, 5, 6]


# --- calculate_statistics (already correct) ---

def test_statistics_basic():
    stats = calculate_statistics([1, 2, 3, 4, 5])
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["median"] == 3


def test_statistics_empty():
    stats = calculate_statistics([])
    assert stats["mean"] == 0
''',
}
