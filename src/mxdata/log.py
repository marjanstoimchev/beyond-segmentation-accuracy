"""Logging setup. Library modules log through ``logging.getLogger(__name__)`` and stay silent until
:func:`setup_logging` is called (the CLI does; notebooks may)."""
from __future__ import annotations

import logging
import sys


class _StderrHandler(logging.StreamHandler):
    """Resolves ``sys.stderr`` at emit time, so a stream swapped later (pytest capture, notebooks) is
    never written to after it was closed."""

    def __init__(self):
        super().__init__(sys.stderr)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):          # StreamHandler.__init__ assigns; ignore
        pass


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger("mxdata")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(isinstance(h, _StderrHandler) for h in root.handlers):
        h = _StderrHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(h)
    root.propagate = False
