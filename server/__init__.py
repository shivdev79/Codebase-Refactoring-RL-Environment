# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Server sub-package for the Codebase Refactoring RL Environment.

Lazy imports are used so that this package works both when installed
as `rlproj.server` (relative imports) and when run as a standalone
script with `from server.xxx import ...` (absolute imports).
"""


def __getattr__(name):
    """Lazy-load exports to avoid import-order issues."""
    if name == "CodeRefactorEnvironment":
        from .rlproj_environment import CodeRefactorEnvironment
        return CodeRefactorEnvironment
    if name == "app":
        from .app import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CodeRefactorEnvironment", "app"]
