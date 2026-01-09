# -*- coding: utf-8 -*-
"""Scrollable frame widget with mouse-wheel support.

Extracted from gui.tabs.motion_analysis_tab to improve code organization.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """A vertical scrollable frame (Canvas + inner frame) with mouse-wheel support."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _on_inner_configure(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_configure(event):
            # keep inner width synced to canvas width
            try:
                self.canvas.itemconfig(self._win, width=event.width)
            except Exception:
                pass

        self.inner.bind("<Configure>", _on_inner_configure)
        self.canvas.bind("<Configure>", _on_canvas_configure)

        self._bind_mousewheel(self.canvas)

    def _bind_mousewheel(self, widget):
        # Windows / macOS
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        # Linux
        widget.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        widget.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _on_mousewheel(self, event):
        # Only scroll if the mouse is over this widget tree
        try:
            x, y = self.winfo_pointerxy()
            w = self.winfo_containing(x, y)
            if w is None:
                return
            if not (w == self.canvas or str(w).startswith(str(self.canvas)) or str(w).startswith(str(self.inner))):
                return
        except Exception:
            pass
        delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        try:
            x, y = self.winfo_pointerxy()
            w = self.winfo_containing(x, y)
            if w is None:
                return
            if not (w == self.canvas or str(w).startswith(str(self.canvas)) or str(w).startswith(str(self.inner))):
                return
        except Exception:
            pass
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

