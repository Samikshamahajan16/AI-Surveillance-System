# ui/app_presentation.py
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
from queue import Queue, Empty

# ensure ui folder import
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.append(HERE)

from surveillance_core import DetectionCore

# ---------- CONFIG ----------
MODEL_NAME = "models/catboost_behavior_full.joblib"
SCALER_NAME = "models/catboost_scaler.joblib"
ALERT_SNAP_FOLDER = "alerts"
LOG_FILE = "alerts/events.log"

VIDEO_W, VIDEO_H = 900, 500
TIMELINE_THUMB_W, TIMELINE_THUMB_H = 120, 70
MAX_TIMELINE = 14

# ---------- COLORS ----------
BG = "#0f1720"
PANEL = "#12161b"
ACCENT = "#10b3d5"
CARD = "#1a2026"
TEXT = "#d7e6ef"
MUTED = "#9aa7b2"
ALERT_RED = "#e23b3b"
OK_GREEN = "#27ae60"


class PresentationUI:
    def __init__(self, root):
        self.root = root
        root.title("AI Surveillance — Presentation Mode")
        root.configure(bg=BG)
        root.geometry("1300x720")

        # Core
        self.core = DetectionCore(
            model_path=MODEL_NAME,
            scaler_path=SCALER_NAME,
            alert_folder=ALERT_SNAP_FOLDER,
            log_file=LOG_FILE
        )
        self.core.on_alert = self._on_alert_callback

        # State
        self.running = False
        self.flashing = False
        self.flash_state = False
        self.last_flash_time = 0
        self.alert_queue = Queue()
        self.timeline = []

        self._build_ui()

        self.update_interval = 30
        self.root.after(self.update_interval, self._update_frame)
        self.root.after(100, self._process_alert_queue)

    # --------------------------------------------------------
    # BUILD UI (FULLY FIXED)
    # --------------------------------------------------------
    def _build_ui(self):

        # ---------- LEFT SIDEBAR ----------
        sidebar = tk.Frame(self.root, bg=PANEL, width=240)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="AI SURVEILLANCE",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 16, "bold")
        ).pack(padx=15, pady=(15, 10))

        tk.Label(
            sidebar,
            text="Activity Log",
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=15, pady=(10, 5))

        # Activity log container
        self.activity_canvas = tk.Canvas(sidebar, bg=PANEL, highlightthickness=0)
        self.activity_canvas.pack(fill=tk.BOTH, expand=True, padx=10)

        self.activity_inner = tk.Frame(self.activity_canvas, bg=PANEL)
        self.activity_canvas.create_window((0, 0), window=self.activity_inner, anchor="nw")

        # Scroll for Activity Log
        self.activity_inner.bind(
            "<Configure>",
            lambda e: self.activity_canvas.configure(
                scrollregion=self.activity_canvas.bbox("all")
            )
        )

        scroll_y = tk.Scrollbar(
            sidebar,
            orient=tk.VERTICAL,
            command=self.activity_canvas.yview
        )
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.activity_canvas.configure(yscrollcommand=scroll_y.set)

        # ---------- MAIN AREA ----------
        main = tk.Frame(self.root, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Frame(main, bg=BG)
        header.pack(fill=tk.X, padx=12, pady=(12, 6))
        tk.Label(
            header,
            text="Live Monitoring",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 18, "bold")
        ).pack(side=tk.LEFT)
        self.status_lbl = tk.Label(header, text="Status: Stopped", bg=BG, fg=MUTED)
        self.status_lbl.pack(side=tk.RIGHT)

        # ---------- VIDEO CARD ----------
        video_card = tk.Frame(main, bg=CARD)
        video_card.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self.canvas = tk.Canvas(
            video_card,
            width=VIDEO_W,
            height=VIDEO_H,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)

        # static canvas items
        self._video_tk = None
        self.video_item = self.canvas.create_image(0, 0, anchor="nw", image=None)

        self.behavior_text_id = self.canvas.create_text(
            20, VIDEO_H - 20,
            anchor="w",
            text="Behavior: --",
            fill="yellow",
            font=("Segoe UI", 14, "bold")
        )

        # Flash overlay
        self.flash_rect = self.canvas.create_rectangle(
            0, 0, VIDEO_W, 70,
            outline="", fill=ALERT_RED,
            state="hidden"
        )
        self.flash_text = self.canvas.create_text(
            VIDEO_W // 2, 35,
            text="!!! Unusual Activity Detected  !!!",
            fill="white",
            font=("Segoe UI", 20, "bold"),
            state="hidden"
        )

        # ---------- CONTROLS ----------
        controls = tk.Frame(video_card, bg=CARD)
        controls.pack(fill=tk.X, pady=(0, 10))

        self.btn_start = ttk.Button(controls, text="Start Camera", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=10)

        self.btn_stop = ttk.Button(controls, text="Stop", state=tk.DISABLED, command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        ttk.Button(controls, text="Snapshots", command=self.open_snapshots).pack(
            side=tk.RIGHT, padx=10
        )

        # ---------- TIMELINE ----------
        tl_frame = tk.Frame(main, bg=BG)
        tl_frame.pack(fill=tk.X, padx=12, pady=(5, 12))

        self.timeline_canvas = tk.Canvas(
            tl_frame, bg=BG,
            height=TIMELINE_THUMB_H + 30,
            highlightthickness=0
        )
        self.timeline_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.timeline_inner = tk.Frame(self.timeline_canvas, bg=BG)
        self.timeline_canvas.create_window((0, 0), window=self.timeline_inner, anchor="nw")

        scrollbar = tk.Scrollbar(tl_frame, orient=tk.HORIZONTAL, command=self.timeline_canvas.xview)
        scrollbar.pack(fill=tk.X)
        self.timeline_canvas.configure(xscrollcommand=scrollbar.set)

    # --------------------------------------------------------
    # START / STOP CAMERA
    # --------------------------------------------------------
    def start(self):
        self.core.start()
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_lbl.config(text="Status: Running", fg=OK_GREEN)
        self.add_log("Camera started", OK_GREEN)

    def stop(self):
        self.core.stop()
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_lbl.config(text="Status: Stopped", fg=MUTED)
        self.add_log("Camera stopped", MUTED)

    # --------------------------------------------------------
    # FRAME UPDATE LOOP
    # --------------------------------------------------------
    def _update_frame(self):
        if self.running:
            frame = self.core.get_frame()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H))
                tkimg = ImageTk.PhotoImage(pil)

                self._video_tk = tkimg
                self.canvas.itemconfig(self.video_item, image=tkimg)

                behavior = getattr(self.core, "current_label", "--")
                self.canvas.itemconfig(self.behavior_text_id, text=f"Behavior: {behavior}")

        # Flash animation
        if self.flashing:
            now = time.time()
            if now - self.last_flash_time > 0.25:
                self.flash_state = not self.flash_state
                self.last_flash_time = now

            if self.flash_state:
                self.canvas.itemconfigure(self.flash_rect, state="normal")
                self.canvas.itemconfigure(self.flash_text, state="normal")
            else:
                self.canvas.itemconfigure(self.flash_rect, state="hidden")
                self.canvas.itemconfigure(self.flash_text, state="hidden")
        else:
            self.canvas.itemconfigure(self.flash_rect, state="hidden")
            self.canvas.itemconfigure(self.flash_text, state="hidden")

        self.root.after(self.update_interval, self._update_frame)

    # --------------------------------------------------------
    # ALERT CALLBACK
    # --------------------------------------------------------
    def _on_alert_callback(self, frame_small_bgr, label, timestamp):
        self.alert_queue.put((frame_small_bgr.copy(), label, timestamp))

    def _process_alert_queue(self):
        try:
            frame_small_bgr, label, timestamp = self.alert_queue.get_nowait()
        except Empty:
            self.root.after(100, self._process_alert_queue)
            return

        # Timeline
        rgb = cv2.cvtColor(frame_small_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        thumb = pil.copy()
        thumb.thumbnail((TIMELINE_THUMB_W, TIMELINE_THUMB_H))

        self.timeline.append({"timestamp": timestamp, "label": label, "thumbnail": thumb})
        if len(self.timeline) > MAX_TIMELINE:
            self.timeline.pop(0)

        self._refresh_timeline()

        # Flash
        self.flashing = True
        self.flash_state = True
        self.last_flash_time = time.time()
        self.root.after(3000, self._stop_flash)

        # Log
        self.add_log(f"{label} detected — snapshot saved", ALERT_RED)

        self.root.after(100, self._process_alert_queue)

    def _stop_flash(self):
        self.flashing = False
        self.flash_state = False

    # --------------------------------------------------------
    # TIMELINE RENDER
    # --------------------------------------------------------
    def _refresh_timeline(self):
        for w in self.timeline_inner.winfo_children():
            w.destroy()

        for item in reversed(self.timeline):
            f = tk.Frame(self.timeline_inner, bg=BG)
            f.pack(side=tk.LEFT, padx=4)

            tkimg = ImageTk.PhotoImage(item["thumbnail"])
            lbl = tk.Label(f, image=tkimg, cursor="hand2", bd=2, relief="ridge")
            lbl.image = tkimg
            lbl.pack()
            lbl.bind("<Button-1>", lambda e, it=item: self._open_snapshot_popup(it))

            ttk.Label(f, text=item["label"], background=BG, foreground=TEXT,
                      font=("Segoe UI", 9, "bold")).pack()
            ttk.Label(f, text=item["timestamp"].split("_")[-1], background=BG,
                      foreground=MUTED, font=("Segoe UI", 8)).pack()

    # --------------------------------------------------------
    # SNAPSHOT POPUP
    # --------------------------------------------------------
    def _open_snapshot_popup(self, item):
        win = tk.Toplevel(self.root)
        win.title(f"{item['label']} — {item['timestamp']}")

        tkimg = ImageTk.PhotoImage(item["thumbnail"])
        lbl = tk.Label(win, image=tkimg)
        lbl.image = tkimg
        lbl.pack(pady=8)

    # --------------------------------------------------------
    # ADD ACTIVITY LOG CARD
    # --------------------------------------------------------
    def add_log(self, text, color):
        card = tk.Frame(self.activity_inner, bg=CARD, bd=1, relief="ridge")
        card.pack(fill=tk.X, pady=4, padx=5)

        tk.Label(card, text=time.strftime("%H:%M:%S"),
                 bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=6, pady=(4, 0))

        tk.Label(card, text=text, bg=CARD, fg=color,
                 font=("Segoe UI", 11, "bold"), wraplength=180, justify="left").pack(
            anchor="w", padx=6, pady=(0, 6)
        )

    # --------------------------------------------------------
    # SNAPSHOT FOLDER
    # --------------------------------------------------------
    def open_snapshots(self):
        path = os.path.abspath(ALERT_SNAP_FOLDER)
        os.makedirs(path, exist_ok=True)
        os.startfile(path)

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------
    def shutdown(self):
        self.running = False
        try:
            self.core.stop()
        except:
            pass


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PresentationUI(root)

    def on_close():
        if messagebox.askokcancel("Quit", "Stop and exit?"):
            app.shutdown()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
