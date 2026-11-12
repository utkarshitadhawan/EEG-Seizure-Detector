"""EEG Seizure Detector - desktop application.      Run:  python gui.py      (or double-click START_HERE.bat)

Two tabs:
  Analyse      open a recording (EDF/BDF/CSV/TXT/NPY) or an EEG image (PNG/JPG), see the verdict, plot and event table, export results.
  Information  "How to use and input formats", then "Training evidence" (graphs and numbers from training).

All analysis logic lives in the `eegseizure` package; this file only builds the window and calls it.
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import matplotlib
from PIL import Image, ImageTk

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk      # noqa: E402
from matplotlib.figure import Figure                                                        # noqa: E402

import eegseizure as z                                                                      # noqa: E402
from eegseizure import minimarkdown, report                                                 # noqa: E402

BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))                 # folder that holds models/, docs/, samples/ ...
APP, VERSION = report.APP_NAME, z.__version__
ALL_CHANNELS, AUTO = "All EEG channels", "Auto"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
MODEL_SEGMENT, MODEL_MULTI = "Single-channel segments (Bonn model)", "Multichannel 18-ch (CHB-MIT, experimental)"
IDLE_STATUS = "Choose an input type, select a file, then press Analyse."


def open_path(path):
    """Open a file or folder with the operating system's default program."""
    if sys.platform.startswith("win"):
        os.startfile(path)                                                                 # noqa: S606 (Windows only)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


class MarkdownView(ttk.Frame):
    """Read-only scrollable view that renders the Markdown subset of eegseizure.minimarkdown (headings, bullets, tables, images)."""
    MAX_IMAGE_WIDTH = 820
    TABLE_WIDTH = 800

    def __init__(self, parent, md_path, base_dir):
        super().__init__(parent)
        self.base_dir, self.images = base_dir, []                                         # keep image references alive
        base = tkfont.nametofont("TkDefaultFont")
        self.family, self.size = base.cget("family"), base.cget("size")
        self.body_font = tkfont.Font(family=self.family, size=self.size)
        self.bold_font = tkfont.Font(family=self.family, size=self.size, weight="bold")
        self.text = tk.Text(self, wrap="word", padx=14, pady=10, relief="flat", cursor="arrow", font=self.body_font)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        for tag, extra in {"h1": 8, "h2": 4, "h3": 2}.items():
            self.text.tag_configure(tag, font=tkfont.Font(family=self.family, size=self.size + extra, weight="bold"), spacing1=10, spacing3=4)
        self.text.tag_configure("bold", font=self.bold_font)
        self.text.tag_configure("code", font=tkfont.nametofont("TkFixedFont"), background="#f0f0f0")
        self.text.tag_configure("bullet", lmargin1=18, lmargin2=34)
        if os.path.isfile(md_path):
            self.render(minimarkdown.parse(open(md_path, encoding="utf-8").read()))
        else:
            self.text.insert("end", f"{os.path.basename(md_path)} was not found next to the program.")
        self.text.configure(state="disabled")

    def inline(self, text, *tags):
        """Insert text with **bold** and `code` pieces styled."""
        for kind, piece in minimarkdown.split_inline(text):
            self.text.insert("end", piece, tuple(tags) + (() if kind == "text" else (kind,)))

    def scroll(self, event):
        """Mouse wheel over a table cell scrolls the page (labels would otherwise swallow the event)."""
        up = event.num == 4 or getattr(event, "delta", 0) > 0
        self.text.yview_scroll(-3 if up else 3, "units")

    def table(self, rows):
        """Draw a table as a grid of wrapping labels (readable even with long cell text)."""
        plain = lambda s: s.replace("**", "").replace("`", "")
        frame = tk.Frame(self.text, bg="#9a9a9a")
        columns = len(rows[0])
        share = [min(max(max(len(plain(r[c])) if c < len(r) else 0 for r in rows), 8), 55) for c in range(columns)]
        for n, row in enumerate(rows):
            for c in range(columns):
                label = tk.Label(frame, text=plain(row[c]) if c < len(row) else "", justify="left", anchor="nw", padx=6, pady=3,
                                 wraplength=int(self.TABLE_WIDTH * share[c] / sum(share)) - 14,
                                 bg="#e6e6e6" if n == 0 else "white", font=self.bold_font if n == 0 else self.body_font)
                label.grid(row=n, column=c, sticky="nsew", padx=(0, 1), pady=(0, 1))
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    label.bind(sequence, self.scroll)
        self.text.window_create("end", window=frame)
        self.text.insert("end", "\n\n")

    def render(self, blocks):
        """Insert all parsed Markdown blocks into the text widget."""
        previous = None
        for kind, value in blocks:
            if previous == "li" and kind != "li":
                self.text.insert("end", "\n")
            previous = kind
            if kind in ("h1", "h2", "h3"):
                self.inline(value, kind)
                self.text.insert("end", "\n")
            elif kind == "p":
                self.inline(value)
                self.text.insert("end", "\n\n")
            elif kind == "li":
                self.text.insert("end", "\u2022 ", "bullet")
                self.inline(value, "bullet")
                self.text.insert("end", "\n")
            elif kind == "code":
                self.text.insert("end", value + "\n\n", "code")
            elif kind == "table":
                self.table(value)
            elif kind == "img":
                alt, rel = value
                path = os.path.join(self.base_dir, rel)
                if os.path.isfile(path):
                    img = Image.open(path)
                    img.thumbnail((self.MAX_IMAGE_WIDTH, 10_000))
                    self.images.append(ImageTk.PhotoImage(img))
                    self.text.image_create("end", image=self.images[-1])
                    self.text.insert("end", "\n\n")


class App(tk.Tk):
    """Main window. `models` is the dict returned by eegseizure.load_models."""

    def __init__(self, models, preload=None):
        super().__init__()
        self.M, self.res, self.summary = models, None, None
        self.q = queue.Queue()                                                              # worker thread -> GUI messages
        self.title(f"{APP} v{VERSION}")
        self.geometry("1180x860")
        self.minsize(980, 720)
        # settings variables
        self.mode = tk.StringVar(value="rec")                                               # "rec" or "img"
        self.path, self.channel, self.fs = tk.StringVar(), tk.StringVar(value=AUTO), tk.StringVar(value=str(z.FS))
        self.model_sel, self.band_pass = tk.StringVar(), tk.BooleanVar(value=True)
        self.seconds, self.lanes, self.uv = tk.StringVar(value="10"), tk.IntVar(value=0), tk.StringVar(value="")
        self.trim_left, self.trim_right = tk.IntVar(value=0), tk.IntVar(value=0)
        self.thr, self.kmin = tk.DoubleVar(value=0.5), tk.IntVar(value=1)
        self.status = tk.StringVar(value=IDLE_STATUS)
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.tab_main, self.tab_info = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(self.tab_main, text="  Analyse  ")
        self.tabs.add(self.tab_info, text="  Information  ")
        tk.Label(self, fg="#a33", font=("Helvetica", 8), text=report.DISCLAIMER).pack(pady=(2, 6))
        self.build_analyse_tab(self.tab_main)
        self.build_info_tab(self.tab_info)
        self.switch_mode()
        self.draw_placeholder()
        self.after(100, self.poll)
        for err in models.get("errors", []):
            messagebox.showwarning(APP, "A model file could not be loaded:\n" + err)
        if preload and os.path.isfile(preload):
            self.mode.set("img" if preload.lower().endswith(IMAGE_EXT) else "rec")
            self.switch_mode()
            self.set_file(preload)

    # ============================================================================== Analyse tab: layout
    def build_analyse_tab(self, tab):
        """Widgets of the Analyse tab: input, settings, verdict, plot, event table, export buttons."""
        top = ttk.LabelFrame(tab, text=" 1. Input ", padding=8)
        top.pack(fill="x", padx=10, pady=(10, 4))
        row = ttk.Frame(top)
        row.pack(fill="x")
        ttk.Radiobutton(row, text="Recording file (EDF / BDF / CSV / TXT / NPY)  - professional", value="rec",
                        variable=self.mode, command=self.switch_mode).pack(side="left")
        ttk.Radiobutton(row, text="Image of an EEG (PNG / JPG)  - quick screening", value="img",
                        variable=self.mode, command=self.switch_mode).pack(side="left", padx=20)
        row = ttk.Frame(top)
        row.pack(fill="x", pady=(6, 0))
        ttk.Entry(row, textvariable=self.path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self.browse).pack(side="left", padx=(6, 0))

        settings = ttk.LabelFrame(tab, text=" 2. Settings ", padding=8)
        settings.pack(fill="x", padx=10, pady=4)
        self.rec_opts = self.build_recording_options(settings)
        self.img_opts = self.build_image_options(settings)
        bottom = ttk.Frame(settings)
        bottom.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(bottom, text="Decision threshold:").pack(side="left")
        ttk.Scale(bottom, from_=0.05, to=0.99, variable=self.thr, length=220, command=lambda v: self.refresh()).pack(side="left", padx=6)
        self.thr_label = ttk.Label(bottom, text="0.50", width=5)
        self.thr_label.pack(side="left")
        ttk.Label(bottom, text="  Min. consecutive windows:").pack(side="left")
        ttk.Spinbox(bottom, from_=1, to=50, textvariable=self.kmin, width=5, command=self.refresh).pack(side="left", padx=4)
        self.kmin.trace_add("write", lambda *a: self.refresh())
        self.analyse_button = ttk.Button(bottom, text="Analyse", command=self.start)
        self.analyse_button.pack(side="right")

        self.verdict = tk.Label(tab, text="NO FILE ANALYSED", font=("Helvetica", 20, "bold"), bg="#e0e0e0", fg="#444", pady=10)
        self.verdict.pack(fill="x", padx=10, pady=4)
        self.stats = ttk.Label(tab, text="", justify="left")
        self.stats.pack(fill="x", padx=12)
        middle = ttk.Frame(tab)
        middle.pack(fill="both", expand=True, padx=10, pady=4)
        self.fig = Figure(figsize=(9, 4.8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=middle)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = ttk.Frame(middle)
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, toolbar).update()
        self.tree = ttk.Treeview(tab, columns=report.EVENT_COLUMNS, show="headings", height=4)
        for col in report.EVENT_COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="w" if col == "Main drivers" else "center", width=420 if col == "Main drivers" else 100)
        self.tree.pack(fill="x", padx=10, pady=4)
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=10, pady=(2, 4))
        for label, command in (("Export CSV", self.export_csv), ("Export report (HTML)", self.export_html),
                               ("Save figure", self.save_figure), ("Clear", self.clear)):
            ttk.Button(bar, text=label, command=command).pack(side="left", padx=(0, 6))
        self.progress = ttk.Progressbar(bar, length=220)
        self.progress.pack(side="right")
        ttk.Label(bar, textvariable=self.status).pack(side="right", padx=10)

    def build_recording_options(self, parent):
        """Settings shown in recording mode: model, channel, sampling rate, band-pass."""
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Model:").grid(row=0, column=0, sticky="w")
        available = [name for name, key in ((MODEL_SEGMENT, "seg"), (MODEL_MULTI, "mc")) if self.M.get(key)]
        self.model_box = ttk.Combobox(frame, textvariable=self.model_sel, values=available, state="readonly", width=38)
        self.model_box.grid(row=0, column=1, padx=(4, 14))
        self.model_sel.set(available[0] if available else "(no model found)")
        self.model_box.bind("<<ComboboxSelected>>", lambda e: self.apply_model_defaults())
        ttk.Label(frame, text="Channel:").grid(row=0, column=2, sticky="w")
        self.channel_box = ttk.Combobox(frame, textvariable=self.channel, values=[AUTO], state="readonly", width=18)
        self.channel_box.grid(row=0, column=3, padx=(4, 14))
        ttk.Label(frame, text="Sampling rate (Hz, text files):").grid(row=0, column=4, sticky="w")
        ttk.Entry(frame, textvariable=self.fs, width=8).grid(row=0, column=5, padx=(4, 14))
        ttk.Checkbutton(frame, text="Band-pass 0.5-40 Hz", variable=self.band_pass).grid(row=0, column=6)
        return frame

    def build_image_options(self, parent):
        """Settings shown in image mode: seconds across the width, channels, trimming, uV per pixel."""
        frame = ttk.Frame(parent)
        ttk.Label(frame, text="Seconds shown across the image width:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.seconds, width=7).grid(row=0, column=1, padx=(4, 14))
        ttk.Label(frame, text="Channels in image (0 = auto):").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(frame, from_=0, to=40, textvariable=self.lanes, width=5).grid(row=0, column=3, padx=(4, 14))
        ttk.Label(frame, text="Trim left/right %:").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(frame, from_=0, to=40, textvariable=self.trim_left, width=4).grid(row=0, column=5, padx=2)
        ttk.Spinbox(frame, from_=0, to=40, textvariable=self.trim_right, width=4).grid(row=0, column=6, padx=2)
        ttk.Label(frame, text="uV per pixel (optional):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frame, textvariable=self.uv, width=7).grid(row=1, column=1, padx=(4, 14), pady=(6, 0))
        ttk.Label(frame, text="Fill this in (from the scale bar) for ACCURATE mode. Blank = uncalibrated, low reliability.",
                  foreground="#555").grid(row=1, column=2, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Label(frame, text="Tip: use a clean screenshot/scan and trim channel labels. Example: a 50 uV bar that is 25 px tall = 2 uV per pixel.",
                  foreground="#555").grid(row=2, column=0, columnspan=7, sticky="w", pady=(2, 0))
        return frame

    def switch_mode(self):
        """Show the settings that belong to the selected input type."""
        self.rec_opts.pack_forget()
        self.img_opts.pack_forget()
        (self.rec_opts if self.mode.get() == "rec" else self.img_opts).pack(side="top", fill="x")
        self.apply_model_defaults()

    def apply_model_defaults(self):
        """The multichannel model works best with 3 consecutive windows and its own threshold; the other modes start at 1 and 0.5."""
        multi = self.mode.get() == "rec" and self.model_sel.get() == MODEL_MULTI and self.M.get("mc")
        self.kmin.set(3 if multi else 1)
        self.thr.set(round(float(self.M["mc"].get("thr_det", 0.5)), 2) if multi else 0.5)

    # ================================================================================ Information tab
    def build_info_tab(self, tab):
        """Information tab with two sub-tabs: user guide and training evidence."""
        inner = ttk.Notebook(tab)
        inner.pack(fill="both", expand=True, padx=6, pady=6)
        guide = ttk.Frame(inner)
        evidence = ttk.Frame(inner)
        inner.add(guide, text="  1. How to use and input formats  ")
        inner.add(evidence, text="  2. Training evidence (graphs and proof)  ")
        self.add_buttons(guide, (("Try a recording example", lambda: self.load_example("rec")),
                                 ("Try an image example", lambda: self.load_example("img")),
                                 ("Open sample files folder", lambda: self.open_if_exists(os.path.join(BASE, "samples")))))
        MarkdownView(guide, os.path.join(BASE, "docs", "USER_GUIDE.md"), BASE).pack(fill="both", expand=True)
        self.add_buttons(evidence, (("Open full training report in browser", self.open_report),
                                    ("Open results folder", lambda: self.open_if_exists(os.path.join(BASE, "results")))))
        MarkdownView(evidence, os.path.join(BASE, "TRAINING_REPORT.md"), BASE).pack(fill="both", expand=True)

    @staticmethod
    def add_buttons(parent, items):
        """Row of buttons above a sub-tab."""
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        for label, command in items:
            ttk.Button(bar, text=label, command=command).pack(side="left", padx=(0, 8))

    def open_if_exists(self, path):
        """Open a file or folder in the system's default program, or tell the user it is missing."""
        if os.path.exists(path):
            open_path(path)
        else:
            messagebox.showinfo(APP, f"{os.path.basename(path)} was not found next to the program.")

    def open_report(self):
        """Open TRAINING_REPORT.html in the web browser."""
        html = Path(BASE, "TRAINING_REPORT.html")
        if html.is_file():
            webbrowser.open(html.as_uri())
        else:
            messagebox.showinfo(APP, "TRAINING_REPORT.html was not found next to the program.")

    def load_example(self, kind):
        """Fill the Analyse tab with a sample file and the right settings, and switch to it (press Analyse to run)."""
        name = "recording_seizure.csv" if kind == "rec" else "image_4channels_seizure.png"
        path = os.path.join(BASE, "samples", "inputs", name)
        if not os.path.isfile(path):
            return messagebox.showinfo(APP, f"Example file {name} was not found in samples/inputs.")
        self.mode.set(kind)
        self.switch_mode()
        if kind == "img":
            self.seconds.set("10")
            self.uv.set("8")
        self.set_file(path)
        self.tabs.select(self.tab_main)

    # ================================================================================ choosing a file
    def browse(self):
        """File dialog; starts in samples/inputs."""
        types = ([("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")] if self.mode.get() == "img"
                 else [("EEG files", "*.edf *.bdf *.csv *.txt *.npy")]) + [("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select file", filetypes=types, initialdir=os.path.join(BASE, "samples", "inputs"))
        if path:
            self.set_file(path)

    def set_file(self, path):
        """Remember the chosen file; for EDF/BDF also list its channels."""
        self.path.set(path)
        self.channel_box.config(values=[AUTO])
        self.channel.set(AUTO)
        if self.mode.get() == "rec" and path.lower().endswith((".edf", ".bdf")):
            try:
                names = z.edf_channels(path)
                self.channel_box.config(values=[ALL_CHANNELS] + names)
                self.channel.set(ALL_CHANNELS)
                self.status.set(f"{len(names)} channels found.")
            except ImportError:
                messagebox.showerror(APP, "Reading EDF files needs the 'mne' package:\n\npip install mne")
            except Exception as e:
                messagebox.showerror(APP, f"Could not read EDF header:\n{e}")

    # ===================================================================================== analysis
    def start(self):
        """Validate the settings and run the analysis in a background thread (so the window stays responsive)."""
        path, mode = self.path.get().strip(), self.mode.get()
        if not os.path.isfile(path):
            return messagebox.showwarning(APP, "Please select a valid file first.")
        if mode == "img" and not path.lower().endswith(IMAGE_EXT):
            return messagebox.showwarning(APP, "Image mode needs a PNG / JPG / BMP / TIFF file.")
        if mode == "rec" and path.lower().endswith(IMAGE_EXT):
            return messagebox.showwarning(APP, "That is an image - switch to 'Image of an EEG' mode.")
        try:
            opts = dict(fs=float(self.fs.get()), seconds=float(self.seconds.get()), lanes=int(self.lanes.get()),
                        trim_l=int(self.trim_left.get()) / 100, trim_r=int(self.trim_right.get()) / 100,
                        uv=float(self.uv.get()) if self.uv.get().strip() else None)
            assert opts["fs"] > 0 and opts["seconds"] > 0 and (opts["uv"] is None or opts["uv"] > 0)
        except Exception:
            return messagebox.showwarning(APP, "Check the numeric settings (sampling rate, seconds, channels, uV per pixel).")
        multi = mode == "rec" and self.model_sel.get() == MODEL_MULTI
        needed = "img" if mode == "img" else ("mc" if multi else "seg")
        if not self.M.get(needed) or (mode == "img" and opts["uv"] and not self.M.get("seg")):
            return messagebox.showerror(APP, "A required model file is missing from the models folder.")
        self.analyse_button.config(state="disabled")
        self.progress.config(mode="indeterminate")
        self.progress.start(12)
        threading.Thread(target=self.worker, args=(mode, multi, path, self.channel.get(), self.band_pass.get(), opts), daemon=True).start()

    def worker(self, mode, multi, path, channel, band_pass, opts):
        """Runs in a background thread. Results and errors are passed back through the queue."""
        progress = lambda kind, value: self.q.put((kind, value))
        try:
            if mode == "img":
                result = z.analyse_image(Image.open(path).convert("RGB"), self.M["img"], opts["seconds"], opts["lanes"], opts["trim_l"],
                                         opts["trim_r"], uv_per_px=opts["uv"], cp_seg=self.M.get("seg"), cb=progress)
                result["path"] = path
            elif multi:
                result = z.analyse_mc(path, self.M["mc"], progress)
            else:
                result = z.analyse_rec(path, opts["fs"], channel, self.M["seg"], progress, band_pass)
            self.q.put(("done", result))
        except ImportError:
            self.q.put(("error", "Reading EDF files needs the 'mne' package:\n\npip install mne"))
        except Exception as e:
            self.q.put(("error", f"{type(e).__name__}: {e}"))

    def poll(self):
        """Called every 100 ms on the GUI thread: handle messages from the worker."""
        try:
            while True:
                kind, value = self.q.get_nowait()
                if kind == "status":
                    self.status.set(value)
                elif kind == "prog":
                    if str(self.progress.cget("mode")) != "determinate":
                        self.progress.stop()
                        self.progress.config(mode="determinate")
                    self.progress["value"] = value * 100
                elif kind == "done":
                    self.idle()
                    self.res = value
                    self.status.set("Done.")
                    self.refresh()
                elif kind == "error":
                    self.idle()
                    self.status.set("Analysis failed.")
                    messagebox.showerror(APP, value)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def idle(self):
        """Reset the progress bar and re-enable the Analyse button."""
        self.progress.stop()
        self.progress.config(mode="determinate", value=0)
        self.analyse_button.config(state="normal")

    # ====================================================================================== results
    def refresh(self):
        """Re-evaluate the stored probabilities with the current threshold / minimum windows and redraw everything."""
        try:
            thr, kmin = round(float(self.thr.get()), 2), max(1, int(self.kmin.get()))
        except (tk.TclError, ValueError):
            return
        self.thr_label.config(text=f"{thr:.2f}")
        if self.res is None:
            return
        self.summary = s = report.summarise(self.res, thr, kmin)
        self.verdict.config(text=s["verdict"], bg=s["color"], fg="white")
        self.stats.config(text=s["text"])
        self.tree.delete(*self.tree.get_children())
        for row in s["rows"]:
            self.tree.insert("", "end", values=row)
        report.plot_result(self.fig, self.res, thr, s["events"])
        self.canvas.draw_idle()

    def draw_placeholder(self):
        """Empty plot with a hint, shown before the first analysis."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Results will appear here", ha="center", va="center", fontsize=14, color="#888", transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw_idle()

    def current(self):
        """(threshold, minimum consecutive windows) as currently set."""
        return round(float(self.thr.get()), 2), max(1, int(self.kmin.get()))

    def export_csv(self):
        """Save the per-window results as CSV."""
        if self.res is None:
            return messagebox.showinfo(APP, "Analyse a file first.")
        default = os.path.splitext(os.path.basename(self.res["path"]))[0] + "_results.csv"
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile=default)
        if path:
            report.results_table(self.res, *self.current()).to_csv(path, index=False)
            self.status.set(f"Saved {os.path.basename(path)}")

    def export_html(self):
        """Save the self-contained HTML report."""
        if self.res is None:
            return messagebox.showinfo(APP, "Analyse a file first.")
        path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html")], initialfile="eeg_report.html")
        if path:
            open(path, "w", encoding="utf-8").write(report.html_report(self.res, self.fig, *self.current()))
            self.status.set(f"Saved {os.path.basename(path)}")

    def save_figure(self):
        """Save the plot as PNG or PDF."""
        if self.res is None:
            return messagebox.showinfo(APP, "Analyse a file first.")
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path, dpi=200)
            self.status.set(f"Saved {os.path.basename(path)}")

    def clear(self):
        """Forget the current result and reset the window."""
        self.res, self.summary = None, None
        self.path.set("")
        self.channel_box.config(values=[AUTO])
        self.channel.set(AUTO)
        self.verdict.config(text="NO FILE ANALYSED", bg="#e0e0e0", fg="#444")
        self.stats.config(text="")
        self.tree.delete(*self.tree.get_children())
        self.draw_placeholder()
        self.status.set(IDLE_STATUS)


def main():
    """Load the models and open the window."""
    models = z.load_models(BASE)
    if not (models["seg"] or models["img"] or models["mc"]):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(APP, "No model files found. Keep the 'models' folder next to gui.py.\n" + "\n".join(models["errors"]))
        sys.exit(1)
    App(models, sys.argv[1] if len(sys.argv) > 1 else None).mainloop()


if __name__ == "__main__":
    main()
