# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Codebase Refactoring RL Environment package.

An OpenEnv-compatible reinforcement learning environment where an agent
iteratively improves a Python codebase by fixing bugs, passing unit tests,
and optimising code quality.

Public API
----------
    from rlproj.client import CodeRefactorEnv
    from rlproj.models import CodeRefactorAction, CodeRefactorObservation, ActionType
"""

from .client import CodeRefactorEnv
from .models import (
    ActionType,
    CodeRefactorAction,
    CodeRefactorObservation,
    TestResult,
)

__all__ = [
    "CodeRefactorEnv",
    "CodeRefactorAction",
    "CodeRefactorObservation",
    "ActionType",
    "TestResult",
]
