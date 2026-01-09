"""Helper functions for GUI."""

from tkinter import messagebox
from typing import Optional


def popup_info(title: str, msg: str):
    """Show info popup."""
    try:
        messagebox.showinfo(title, msg)
    except Exception:
        pass


def popup_error(title: str, msg: str):
    """Show error popup."""
    try:
        messagebox.showerror(title, msg)
    except Exception:
        pass


def cam_id(i: int) -> str:
    """Pose2Sim-friendly camera id (1-based: cam_01, cam_02, ...)."""
    return f"cam_{i+1:02d}"

