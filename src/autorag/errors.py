"""Error types for AutoRAG's extras model.

Every audio / RAG method on :class:`~autorag.core.AutoRAG` imports its
heavy dependencies inside the method body and re-raises
:class:`ModuleNotFoundError` as :class:`MissingExtraError` with a hint
naming the install extra that fixes it.
"""

from __future__ import annotations


class MissingExtraError(ImportError):
    """Raised when an :class:`~autorag.core.AutoRAG` method needs an
    optional extra that isn't installed."""


def _missing_extra(extras: str, original: BaseException) -> MissingExtraError:
    """Build a :class:`MissingExtraError` that names the install hint."""
    return MissingExtraError(
        f"This feature requires `pip install 'autorag[{extras}]'`. "
        f"Underlying import error: {original}"
    )
