"""Errors that name themselves: a stable code, a plain message, and a hint on what to change."""
from __future__ import annotations


class MxDataError(Exception):
    code = "E_MXDATA"

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message, self.hint = message, hint

    def __str__(self) -> str:
        s = f"[{self.code}] {self.message}"
        return f"{s}\n  hint: {self.hint}" if self.hint else s


class ConfigError(MxDataError):
    code = "E_CONFIG"


class SlideError(MxDataError):
    code = "E_SLIDE"


class AnnotationError(MxDataError):
    code = "E_ANNOTATION"


class ValidationError(MxDataError):
    code = "E_VALIDATION"


class DatasetError(MxDataError):
    code = "E_DATASET"
