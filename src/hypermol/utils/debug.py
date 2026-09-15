"""Small compatibility helpers for code migrated from the research prototype."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def profile(obj: T) -> T:
    """No-op replacement for the prototype's optional profiling decorator."""
    return obj
