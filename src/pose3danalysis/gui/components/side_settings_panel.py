# -*- coding: utf-8 -*-
"""Side settings panel widget that can be collapsed.

Extracted from gui.tabs.motion_analysis_tab to improve code organization.
"""

from __future__ import annotations

from typing import Callable

from tkinter import ttk

from pose3danalysis.gui.components.scrollable_frame import ScrollableFrame


class SideSettingsPanel(ttk.Frame):
    """Right settings panel that can be collapsed sideways (toggle button always visible)."""

    def __init__(self, parent, on_toggle: Callable[[bool], None], *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._on_toggle = on_toggle
        self._open = True

        # Header with toggle button (always visible, never hidden)
        self.header = ttk.Frame(self)
        self.header.pack(fill="x", padx=6, pady=(6, 0))
        # Button with fixed width, right-aligned for better appearance
        self.btn = ttk.Button(self.header, text="◀", command=self.toggle, width=3)
        self.btn.pack(side="right", padx=(0, 0), pady=0)

        # Scrollable body (can be hidden/shown)
        self._scroll = ScrollableFrame(self)
        self._scroll.pack(fill="both", expand=True, padx=6, pady=6)

        self.body = self._scroll.inner

    def toggle(self):
        self._open = not self._open
        self._on_toggle(self._open)

    def set_header_text(self, open_: bool):
        self.btn.configure(text=("◀" if open_ else "▶"))

    def show_body(self):
        # avoid duplicate packing
        if not self._scroll.winfo_ismapped():
            self._scroll.pack(fill="both", expand=True, padx=6, pady=6)

    def hide_body(self):
        if self._scroll.winfo_ismapped():
            self._scroll.pack_forget()
    
    def is_open(self) -> bool:
        """Return whether the panel is currently open."""
        return self._open

