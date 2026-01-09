from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np


class PointPicker:
    """
    - 시작 시 창에 맞게 자동 fit (원본 비율 유지)
    - 마우스 휠: 실제 이미지 리사이즈 기반 줌
    - 우클릭 드래그: pan
    - 좌클릭: 포인트 추가
    - Backspace: 마지막 포인트 제거
    """

    def __init__(self, title: str, rgb: np.ndarray, n_points: int):
        self.title = title
        self.rgb = rgb
        self.n_points = int(n_points)

        self.points: list[tuple[float, float]] = []
        self._ok = False

        self.scale = 1.0
        self.min_scale = 0.05
        self.max_scale = 20.0

        self.offset_x = 0.0
        self.offset_y = 0.0

        self._photo: ImageTk.PhotoImage | None = None
        self._img_id = None
        self._overlay_ids: list[int] = []

        self._pan_last: tuple[int, int] | None = None
        self._did_fit_once = False
        self._photo_scale: float | None = None

    def pick(self) -> np.ndarray | None:
        root = tk.Toplevel()
        root.title(self.title)
        root.geometry("1000x700")
        root.minsize(700, 500)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.status = ttk.Label(root, text="")
        self.status.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        btns = ttk.Frame(root)
        btns.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        btns.columnconfigure(0, weight=1)

        ttk.Button(btns, text="Undo (Backspace)", command=self.undo).pack(side="left")
        ttk.Button(btns, text="Cancel", command=lambda: self._finish(root, ok=False)).pack(side="right")
        ttk.Button(btns, text="OK", command=lambda: self._finish(root, ok=True)).pack(side="right", padx=6)

        self.img_pil = Image.fromarray(self.rgb)
        self.w0, self.h0 = self.img_pil.size

        self.canvas.bind("<Button-1>", self.on_click)

        self.canvas.bind("<Button-3>", self.pan_start)
        self.canvas.bind("<B3-Motion>", self.pan_move)
        self.canvas.bind("<ButtonRelease-3>", self.pan_end)

        self.canvas.bind("<MouseWheel>", self.on_wheel)          # Windows
        self.canvas.bind("<Button-4>", self.on_wheel_linux)      # Linux up
        self.canvas.bind("<Button-5>", self.on_wheel_linux)      # Linux down

        root.bind("<BackSpace>", lambda e: self.undo())
        root.bind("<Escape>", lambda e: self._finish(root, ok=False))

        root.after(50, self._fit_to_canvas)
        self._render(full=True)

        root.grab_set()
        root.wait_window()

        if not self._ok:
            return None
        return np.asarray(self.points, dtype=np.float32)

    def _finish(self, win, ok: bool):
        self._ok = bool(ok) and (len(self.points) == self.n_points)
        if self.n_points == 0 and ok:
            self._ok = True
        win.destroy()

    def _fit_to_canvas(self):
        if self._did_fit_once:
            return
        cw = max(self.canvas.winfo_width(), 2)
        ch = max(self.canvas.winfo_height(), 2)

        pad = 20
        fit_scale = min((cw - pad) / self.w0, (ch - pad) / self.h0)
        if fit_scale <= 0:
            fit_scale = 1.0

        self.scale = max(self.min_scale, min(self.max_scale, fit_scale))

        img_w = self.w0 * self.scale
        img_h = self.h0 * self.scale
        self.offset_x = max((cw - img_w) / 2, 0)
        self.offset_y = max((ch - img_h) / 2, 0)

        self._did_fit_once = True
        self._render(full=True)

    def _render(self, full: bool = False):
        need_new_photo = (self._photo is None) or (self._photo_scale is None) or (abs(self._photo_scale - self.scale) > 1e-6)

        if full or self._img_id is None:
            self.canvas.delete("all")
            self._img_id = None
            self._overlay_ids.clear()
            need_new_photo = True

        if need_new_photo:
            w = max(1, int(round(self.w0 * self.scale)))
            h = max(1, int(round(self.h0 * self.scale)))
            img_disp = self.img_pil.resize((w, h), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(img_disp)
            self._photo_scale = self.scale

            if self._img_id is None:
                self._img_id = self.canvas.create_image(self.offset_x, self.offset_y, image=self._photo, anchor="nw", tags=("img",))
            else:
                self.canvas.itemconfigure(self._img_id, image=self._photo)
                self.canvas.coords(self._img_id, self.offset_x, self.offset_y)
        else:
            if self._img_id is not None:
                self.canvas.coords(self._img_id, self.offset_x, self.offset_y)

        for oid in self._overlay_ids:
            self.canvas.delete(oid)
        self._overlay_ids.clear()

        for i, (x, y) in enumerate(self.points, start=1):
            xs = self.offset_x + x * self.scale
            ys = self.offset_y + y * self.scale
            r = 5
            oid = self.canvas.create_oval(xs - r, ys - r, xs + r, ys + r, outline="red", width=2)
            tid = self.canvas.create_text(xs + 10, ys, text=str(i), fill="yellow", anchor="w", font=("Segoe UI", 10, "bold"))
            self._overlay_ids.extend([oid, tid])

        self.status.config(
            text=f"{len(self.points)}/{self.n_points} points (Left click add, Backspace undo, Wheel zoom, Right-drag pan)"
        )

    def pan_start(self, event):
        self._pan_last = (event.x, event.y)

    def pan_move(self, event):
        if self._pan_last is None:
            return
        lx, ly = self._pan_last
        dx = event.x - lx
        dy = event.y - ly
        self.offset_x += dx
        self.offset_y += dy
        self._pan_last = (event.x, event.y)
        self._render(full=False)

    def pan_end(self, event):
        self._pan_last = None

    def on_wheel(self, event):
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._zoom_at(event.x, event.y, factor)

    def on_wheel_linux(self, event):
        if event.num == 4:
            self._zoom_at(event.x, event.y, 1.15)
        elif event.num == 5:
            self._zoom_at(event.x, event.y, 1 / 1.15)

    def _zoom_at(self, cx: int, cy: int, factor: float):
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        if abs(new_scale - self.scale) < 1e-8:
            return

        img_x = (cx - self.offset_x) / self.scale
        img_y = (cy - self.offset_y) / self.scale

        self.scale = new_scale
        self.offset_x = cx - img_x * self.scale
        self.offset_y = cy - img_y * self.scale

        self._render(full=False)

    def undo(self):
        if self.points:
            self.points.pop()
            self._render(full=False)

    def on_click(self, event):
        if len(self.points) >= self.n_points:
            return

        x_img = (event.x - self.offset_x) / self.scale
        y_img = (event.y - self.offset_y) / self.scale

        if x_img < 0 or y_img < 0 or x_img >= self.w0 or y_img >= self.h0:
            return

        self.points.append((float(x_img), float(y_img)))
        self._render(full=False)

class ImageViewer:
    """
    Zoom/Pan viewer for RGB image
    - Wheel zoom
    - Right-drag pan
    - ESC close
    """
    def __init__(self, title: str, rgb: np.ndarray):
        self.title = title
        self.rgb = rgb.astype(np.uint8)
        self.scale = 1.0
        self.min_scale = 0.05
        self.max_scale = 20.0
        self.off_x = 0.0
        self.off_y = 0.0
        self._pan_last = None
        self._photo = None
        self._img_id = None

    def show(self):
        win = tk.Toplevel()
        win.title(self.title)
        win.geometry("1000x700")
        win.minsize(700, 500)

        self.canvas = tk.Canvas(win, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        info = ttk.Label(win, text="Wheel: zoom | Right-drag: pan | ESC: close")
        info.pack(fill="x")

        self.canvas.bind("<Configure>", lambda e: self._fit_to_canvas())
        self.canvas.bind("<MouseWheel>", self._on_wheel)      # Windows
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel_linux(+1))  # Linux up
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel_linux(-1))  # Linux down
        self.canvas.bind("<Button-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._pan_end)

        win.bind("<Escape>", lambda e: win.destroy())
        self.canvas.focus_set()
        self._fit_to_canvas()

    def _fit_to_canvas(self):
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        ih, iw = self.rgb.shape[:2]
        s = min(w / iw, h / ih)
        self.scale = max(self.min_scale, min(self.max_scale, s))
        self.off_x = (w - iw * self.scale) / 2.0
        self.off_y = (h - ih * self.scale) / 2.0
        self._render()

    def _render(self):
        ih, iw = self.rgb.shape[:2]
        new_w = max(1, int(round(iw * self.scale)))
        new_h = max(1, int(round(ih * self.scale)))

        img = Image.fromarray(self.rgb).resize((new_w, new_h), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self._img_id = self.canvas.create_image(self.off_x, self.off_y, anchor="nw", image=self._photo)

    def _zoom_at(self, cx, cy, factor):
        # cursor 기준으로 zoom
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
        factor = 1.15 if e.delta > 0 else (1 / 1.15)
        self._zoom_at(e.x, e.y, factor)

    def _on_wheel_linux(self, direction: int):
        factor = 1.15 if direction > 0 else (1 / 1.15)
        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2
        self._zoom_at(cx, cy, factor)

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

    def _pan_end(self, e):
        self._pan_last = None