from __future__ import annotations


class MissingExtraError(ImportError):
    """Raised when an `AutoRAG` method needs an optional extra that isn't installed."""


def _missing_extra(extras: str, original: BaseException) -> MissingExtraError:
    return MissingExtraError(
        f"This feature requires `pip install 'autorag[{extras}]'`. "
        f"Underlying import error: {original}"
    )
