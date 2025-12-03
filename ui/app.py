# ui/app.py
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
import os
from surveillance_core import DetectionCore

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "models/catboost_behavior.cbm"
SCALER_NAME = "models/catboost_scaler.joblib"
ALERT_SNAP_FOLDER = "alerts"
LOG_FILE = "alerts/events.log"


class SurveillanceUI:
    def __init__(self, root):
        self.root = root
        root.title("AI Surveillance System")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Core detection engine
        self.core = DetectionCore(
            model_path=MODEL_NAME,
            scaler_path=SCALER_NAME,
            alert_folder=ALERT_SNAP_FOLDER,
            log_file=LOG_FILE
        )

        self.core.on_alert = self.on_alert_callback

        # Timeline
        self.timeline = []

        # Flash state
        self.flashing = False
        self.flash_state = False
        self.last_flash = 0

        # Fight stopped (green indicator)
        self.show_stop_banner = False
        self.stop_banner_start = None

        self.setup_widgets()

        self.update_interval = 30
        self.root.after(self.update_interval, self.update_frame_loop)

    # ----------------------------------------------------
    # UI Layout
    # ----------------------------------------------------
    def setup_widgets(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Video feed canvas
        self.video_w = 960
        self.video_h = 540
        self.canvas = tk.Canvas(top, width=self.video_w, height=self.video_h, bg="black")
        self.canvas.pack(side=tk.LEFT, padx=(0, 6))

        # Right controls
        right = ttk.Frame(top, width=280)
        right.pack(side=tk.LEFT, fill=tk.Y)

        self.btn_start = ttk.Button(right, text="Start Surveillance", command=self.start)
        self.btn_start.pack(fill=tk.X, pady=(5, 5))

        self.btn_stop = ttk.Button(right, text="Stop", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(right, text="Open Snapshot Folder", command=self.open_snapshots)\
            .pack(fill=tk.X, pady=(0, 10))

        # Event log
        ttk.Label(right, text="Event Log:").pack(anchor="w")
        self.log_box = tk.Text(right, width=32, height=22, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")\
            .pack(side=tk.BOTTOM, fill=tk.X)

        # Timeline
        self.timeline_frame = ttk.Frame(self.root)
        self.timeline_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=4)

    # ----------------------------------------------------
    # Surveillance Control
    # ----------------------------------------------------
    def start(self):
        self.core.start()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.append_log("Surveillance started.")
        self.status_var.set("Running")

    def stop(self):
        self.core.stop()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.append_log("Surveillance stopped.")
        self.status_var.set("Stopped")

    # ----------------------------------------------------
    def open_snapshots(self):
        folder = os.path.abspath(ALERT_SNAP_FOLDER)
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder)

    # ----------------------------------------------------
    # Logging
    # ----------------------------------------------------
    def append_log(self, text):
        self.log_box.configure(state=tk.NORMAL)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_box.insert("end", f"{ts} — {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state=tk.DISABLED)

    # ----------------------------------------------------
    # Alert callback
    # ----------------------------------------------------
    def on_alert_callback(self, frame_small, label, timestamp):
        self.append_log(f"{label} detected (snapshot saved).")

        # Save timeline thumbnail
        thumb = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
        self.timeline.append({"timestamp": timestamp, "label": label, "thumbnail": thumb})
        if len(self.timeline) > 20:
            self.timeline.pop(0)
        self.update_timeline()

        # Activate flashing red banner
        self.flashing = True
        self.flash_state = True
        self.last_flash = time.time()

        # Hide stop banner if fight starts again
        self.show_stop_banner = False

    # ----------------------------------------------------
    # Timeline UI
    # ----------------------------------------------------
    def update_timeline(self):
        for w in self.timeline_frame.winfo_children():
            w.destroy()

        for item in self.timeline[-8:][::-1]:
            frame = ttk.Frame(self.timeline_frame, borderwidth=1, relief="solid")
            frame.pack(side=tk.LEFT, padx=3)

            img = item["thumbnail"].resize((80, 60))
            tkimg = ImageTk.PhotoImage(img)

            lbl = tk.Label(frame, image=tkimg, cursor="hand2")
            lbl.image = tkimg
            lbl.pack()

            lbl.bind("<Button-1>", lambda e, it=item: self.open_timeline_image(it))

            ttk.Label(frame, text=item["timestamp"].split("_")[-1], font=("Segoe UI", 7)).pack()
            ttk.Label(frame, text=item["label"], font=("Segoe UI", 8, "bold")).pack()

    # ----------------------------------------------------
    def open_timeline_image(self, item):
        win = tk.Toplevel(self.root)
        win.title(item["label"] + " — " + item["timestamp"])

        img = item["thumbnail"]
        max_w = 700
        scale = max_w / img.width
        big = img.resize((max_w, int(img.height * scale)))

        tkimg = ImageTk.PhotoImage(big)
        lbl = tk.Label(win, image=tkimg)
        lbl.image = tkimg
        lbl.pack()

        ttk.Label(win, text=item["timestamp"]).pack()
        ttk.Label(win, text=item["label"], font=("Segoe UI", 10, "bold")).pack()
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)

    # ----------------------------------------------------
    # Red flash overlay
    # ----------------------------------------------------
    def draw_flash(self, pil_img):
        from PIL import ImageDraw
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        if self.flash_state:
            draw.rectangle((0, 0, w, 70), fill=(200, 0, 0, 180))
            draw.text((20, 20), "!!! FIGHT DETECTED !!!", fill="white")

        return ImageTk.PhotoImage(img)

    # ----------------------------------------------------
    # Main loop
    # ----------------------------------------------------
    def update_frame_loop(self):
        frame = self.core.get_frame()
        label = self.core.current_label

        # Detect fight stop
        if label == "No Fight" and self.flashing:
            self.flashing = False
            self.flash_state = False
            self.show_stop_banner = True
            self.stop_banner_start = time.time()

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((self.video_w, self.video_h))

            # Red flashing
            if self.flashing:
                if time.time() - self.last_flash > 0.3:
                    self.flash_state = not self.flash_state
                    self.last_flash = time.time()
                tkimg = self.draw_flash(pil)
            else:
                tkimg = ImageTk.PhotoImage(pil)

            self.canvas.imgtk = tkimg
            self.canvas.create_image(0, 0, anchor="nw", image=tkimg)

            # Behavior text
            self.canvas.create_text(
                20, 30,
                text=f"Behavior: {label}",
                anchor="w",
                fill="yellow",
                font=("Segoe UI", 18, "bold")
            )

            # Green “Fight Stopped”
            if self.show_stop_banner:
                if time.time() - self.stop_banner_start < 2:
                    self.canvas.create_rectangle(0, 0, self.video_w, 70, fill="#009900", outline="")
                    self.canvas.create_text(
                        self.video_w // 2, 35,
                        text="✔ FIGHT STOPPED",
                        fill="white",
                        font=("Segoe UI", 20, "bold")
                    )
                else:
                    self.show_stop_banner = False

        self.root.after(self.update_interval, self.update_frame_loop)

    # ----------------------------------------------------
    def on_close(self):
        if messagebox.askokcancel("Quit", "Exit the surveillance system?"):
            self.core.stop()
            self.root.destroy()


# ----------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SurveillanceUI(root)
    root.mainloop()
