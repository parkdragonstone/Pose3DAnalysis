import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class ZoomPreview(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg="black", highlightthickness=0, **kwargs)
        self._rgb = None
        self._img = None
        self._photo = None
        self._img_id = None

        self.scale = 1.0
        self.min_scale = 0.05
        self.max_scale = 20.0
        self.off_x = 0.0
        self.off_y = 0.0
        self._pan_last = None

        self.bind("<Configure>", lambda e: self._fit())
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", lambda e: self._on_wheel_linux(+1))
        self.bind("<Button-5>", lambda e: self._on_wheel_linux(-1))

        # pan: middle-click (wheel) and right-click behave the same
        self.bind("<Button-2>", self._pan_start)
        self.bind("<B2-Motion>", self._pan_move)
        self.bind("<ButtonRelease-2>", lambda e: setattr(self, "_pan_last", None))
        self.bind("<Button-3>", self._pan_start)
        self.bind("<B3-Motion>", self._pan_move)
        self.bind("<ButtonRelease-3>", lambda e: setattr(self, "_pan_last", None))

        self.bind("<Double-Button-1>", lambda e: self._fit())
        self.bind("<Enter>", lambda e: self.focus_set())

    def set_rgb(self, rgb: np.ndarray, fit_if_first=True):
        self._rgb = rgb.astype(np.uint8)
        self._img = Image.fromarray(self._rgb)
        if fit_if_first:
            self._fit()
        else:
            self._render()

    def _fit(self):
        if self._img is None:
            return
        cw = max(2, self.winfo_width())
        ch = max(2, self.winfo_height())
        iw, ih = self._img.size
        s = min(cw / iw, ch / ih)
        self.scale = max(self.min_scale, min(self.max_scale, s))
        self.off_x = (cw - iw * self.scale) / 2.0
        self.off_y = (ch - ih * self.scale) / 2.0
        self._render()

    def _render(self):
        if self._img is None:
            return
        iw, ih = self._img.size
        w = max(1, int(round(iw * self.scale)))
        h = max(1, int(round(ih * self.scale)))
        disp = self._img.resize((w, h), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(disp)
        self.delete("all")
        self._img_id = self.create_image(self.off_x, self.off_y, anchor="nw", image=self._photo)

    def _zoom_at(self, cx, cy, factor):
        if self._img is None:
            return
        ix = (cx - self.off_x) / self.scale
        iy = (cy - self.off_y) / self.scale
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        if abs(new_scale - self.scale) < 1e-9:
            return
        self.scale = new_scale
        self.off_x = cx - ix * self.scale
        self.off_y = cy - iy * self.scale
        self._render()

    def _on_wheel(self, e):
        self._zoom_at(e.x, e.y, 1.15 if e.delta > 0 else (1/1.15))

    def _on_wheel_linux(self, direction):
        cx = self.winfo_width() // 2
        cy = self.winfo_height() // 2
        self._zoom_at(cx, cy, 1.15 if direction > 0 else (1/1.15))

    def _pan_start(self, e):
        self._pan_last = (e.x, e.y)

    def _pan_move(self, e):
        if self._pan_last is None:
            return
        dx = e.x - self._pan_last[0]
        dy = e.y - self._pan_last[1]
        self.off_x += dx
        self.off_y += dy
        self._pan_last = (e.x, e.y)
        self._render()
