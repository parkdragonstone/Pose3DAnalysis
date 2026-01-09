"""Logging utilities for GUI applications."""

import logging
from pathlib import Path
from typing import Optional, Callable
import tkinter as tk
from tkinter import scrolledtext


def setup_file_logging(log_path: Path, logger_name: str = "Pose3DAnalysis") -> logging.Logger:
    """Setup file-based logging."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path)
               for h in logger.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_to_ui(log_text: Optional[scrolledtext.ScrolledText], msg: str, root: Optional[tk.Tk] = None):
    """Append message to UI log text widget."""
    if log_text is None:
        return

    def _append():
        try:
            log_text.configure(state="normal")
            log_text.insert("end", msg + "\n")
            log_text.see("end")
            log_text.configure(state="disabled")
        except Exception:
            pass

    if root is not None:
        try:
            root.after(0, _append)
        except Exception:
            _append()
    else:
        _append()


class UILogHandler(logging.Handler):
    """Forward Python `logging` records to the Motion Analysis log box."""

    def __init__(self, emit_fn: Callable[[str], None]):
        super().__init__()
        self._emit_fn = emit_fn

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        try:
            self._emit_fn(msg)
        except Exception:
            pass

