# launch.py
import os, sys, subprocess, threading, tkinter as tk
from tkinter import ttk, messagebox

ROOT = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd, env=None):
    def _run():
        try:
            subprocess.Popen(cmd, cwd=ROOT, env=env or os.environ.copy())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run:\n{' '.join(cmd)}\n\n{e}")
    threading.Thread(target=_run, daemon=True).start()

def ensure_ollama(model):
    # pull (idempotent) and run model in background if not running
    run_cmd(["ollama", "pull", model])
    # try to run (if already running, it will just serve another session)
    run_cmd(["ollama", "run", model])

def build_index():
    btn_build.config(state="disabled")
    model = model_var.get()
    ensure_ollama(model)
    # turn off chroma telemetry for peace of mind
    env = os.environ.copy()
    env["ANONYMIZED_TELEMETRY"] = "FALSE"
    run_cmd([sys.executable, "ingest.py"], env=env)
    btn_build.config(state="normal")
    messagebox.showinfo("RAGTIME", "Index build launched.\n(You can open the UI now.)")

def open_app():
    btn_open.config(state="disabled")
    model = model_var.get()
    ensure_ollama(model)
    env = os.environ.copy()
    env["ANONYMIZED_TELEMETRY"] = "FALSE"
    env["LLM_MODEL"] = model
    run_cmd(["streamlit", "run", "serve.py"], env=env)
    btn_open.config(state="normal")

def open_data_folder():
    path = os.path.join(ROOT, "data", "notes")
    os.makedirs(path, exist_ok=True)
    if sys.platform.startswith("darwin"):
        subprocess.call(["open", path])
    elif os.name == "nt":
        subprocess.call(["explorer", path])
    else:
        subprocess.call(["xdg-open", path])

app = tk.Tk()
app.title("RAGTIME Launcher")

frm = ttk.Frame(app, padding=12)
frm.grid(sticky="nsew")
app.columnconfigure(0, weight=1)
app.rowconfigure(0, weight=1)

ttk.Label(frm, text="Model").grid(row=0, column=0, sticky="w")
model_var = tk.StringVar(value="gemma:2b")
ttk.Combobox(frm, textvariable=model_var, values=["gemma:2b","mistral","llama3:8b"], width=20).grid(row=0, column=1, sticky="ew", padx=6)

ttk.Separator(frm).grid(row=1, columnspan=3, sticky="ew", pady=8)

ttk.Button(frm, text="Open data/notes…", command=open_data_folder).grid(row=2, column=0, sticky="w")
btn_build = ttk.Button(frm, text="Build / Rebuild Index", command=build_index)
btn_build.grid(row=2, column=1, sticky="ew", padx=6)

btn_open = ttk.Button(frm, text="Open RAGTIME (Streamlit)", command=open_app)
btn_open.grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)

ttk.Label(frm, text="Tip: drop PDFs in data/extras/ and .ipynb in data/notes/, then Build.").grid(row=4, column=0, columnspan=2, sticky="w", pady=6)

app.mainloop()


# python3 launch.py
# pip install pyinstaller
# pyinstaller -F launch.py   # or -w for no console window on Win/Mac
