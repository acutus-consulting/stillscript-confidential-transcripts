import os
import sys
import platform
import subprocess
from pathlib import Path
import threading
import customtkinter as ctk
from PIL import Image
import whisper
from tkinter import filedialog, messagebox

# VERSION CONTROL
VERSION = "1.4.0"

# --- Globale model cache ---
_model_cache = {}

def resource_path(relative_path):
    """Kry die absolute pad na bronne, werk vir dev en vir PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def open_folder(path):
    """Maak 'n gids oop op enige bedryfstelsel, veilig."""
    try:
        if not os.path.exists(path):
            return
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showwarning("Waarskuwing", f"Kon nie gids oopmaak nie:\n{e}")

def get_model(model_size):
    """Laai die Whisper model, gebruik cache as dit al gelaai is."""
    if model_size not in _model_cache:
        _model_cache[model_size] = whisper.load_model(model_size)
    return _model_cache[model_size]

def update_ui(func, *args, **kwargs):
    """Stuur UI-opdaterings veilig na die hoofthread."""
    app.after(0, func, *args, **kwargs)

def transcribe_audio():
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg *.mp4")]
    )

    if not file_path:
        return

    task_choice = task_var.get()
    whisper_task = "translate" if task_choice == "Translate to English" else "transcribe"
    model_size = model_var.get()
    language = language_var.get()
    lang_code = None if language == "Auto Detect" else language

    button.configure(state="disabled")
    update_ui(progress_bar.set, 0.05)
    update_ui(status_label.configure, text=f"Status: Initializing ({task_choice})...")

    def run():
        try:
            update_ui(status_label.configure, text="Status: Loading AI Model (Internet required for first run)...")
            update_ui(progress_bar.set, 0.2)

            model = get_model(model_size)

            update_ui(progress_bar.set, 0.4)
            update_ui(status_label.configure, text="Status: Analyzing Audio... please wait.")

            prompt = "Hierdie is Afrikaanse spraak." if lang_code == "af" else None
            result = model.transcribe(file_path, task=whisper_task, language=lang_code, initial_prompt=prompt)

            update_ui(progress_bar.set, 0.8)
            update_ui(status_label.configure, text="Status: Saving file...")

            # Bou die uitvoerpad stap vir stap
            home = Path.home()
            downloads_path = home / "Downloads" / "DanScribe_Transcriptions"
            downloads_path.mkdir(parents=True, exist_ok=True)

            file_name = os.path.basename(file_path)
            suffix = "_translated" if whisper_task == "translate" else "_original"
            output_file_name = os.path.splitext(file_name)[0] + suffix + ".txt"
            final_output_path = downloads_path / output_file_name

            with open(str(final_output_path), "w", encoding="utf-8") as f:
                f.write(result["text"])

            update_ui(progress_bar.set, 1.0)
            update_ui(status_label.configure, text="Status: Success!")

            messagebox.showinfo("DanScribe AI", f"Done!\nSaved to:\n{final_output_path}")

            open_folder(str(downloads_path))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            update_ui(status_label.configure, text="Status: Error — see message box")

        finally:
            update_ui(button.configure, state="normal")
            update_ui(progress_bar.set, 0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

# --- UI Setup ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title(f"DanScribe AI - Professional Edition v{VERSION}")
app.geometry("600x780")
app.resizable(False, False)

# LOGO
try:
    logo_full_path = resource_path("logo.jpg")
    img = Image.open(logo_full_path)
    logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(180, 180))
    logo_label = ctk.CTkLabel(app, image=logo_img, text="")
    logo_label.pack(pady=20)
except Exception:
    header = ctk.CTkLabel(app, text="DanScribe AI", font=("Arial", 32, "bold"), text_color="#3b8ed0")
    header.pack(pady=20)

# MODEL KEUSE
model_label = ctk.CTkLabel(app, text="Select AI Model:", font=("Arial", 14, "bold"))
model_label.pack(pady=(10, 2))

model_var = ctk.StringVar(value="base")
model_menu = ctk.CTkOptionMenu(
    app,
    values=["tiny", "base", "small", "medium"],
    variable=model_var,
    width=250
)
model_menu.pack(pady=(0, 5))

model_hint = ctk.CTkLabel(
    app,
    text="tiny = fastest  |  medium = most accurate",
    font=("Arial", 11),
    text_color="gray"
)
model_hint.pack()

# TAAL KEUSE
lang_label = ctk.CTkLabel(app, text="Audio Language:", font=("Arial", 14, "bold"))
lang_label.pack(pady=(15, 2))

language_var = ctk.StringVar(value="af")
lang_menu = ctk.CTkOptionMenu(
    app,
    values=["Auto Detect", "af", "en", "nl", "de", "fr", "es", "pt", "zu", "xh"],
    variable=language_var,
    width=250
)
lang_menu.pack(pady=(0, 5))

lang_hint = ctk.CTkLabel(
    app,
    text="af = Afrikaans  |  en = English  |  nl = Nederlands  |  zu = Zulu  |  xh = Xhosa",
    font=("Arial", 11),
    text_color="gray",
    wraplength=500
)
lang_hint.pack()

# TAAK KEUSE
task_label = ctk.CTkLabel(app, text="Select Mode:", font=("Arial", 14, "bold"))
task_label.pack(pady=(15, 2))

task_var = ctk.StringVar(value="Original Language")
task_menu = ctk.CTkOptionMenu(
    app,
    values=["Original Language", "Translate to English"],
    variable=task_var,
    width=250
)
task_menu.pack(pady=(0, 10))

# STATUS & PROGRESS
status_label = ctk.CTkLabel(app, text=f"Ready | Version {VERSION}", font=("Arial", 14))
status_label.pack(pady=10)

progress_bar = ctk.CTkProgressBar(app, width=450)
progress_bar.pack(pady=10)
progress_bar.set(0)

# START KNOPPIE
button = ctk.CTkButton(
    app,
    text="SELECT FILE & START",
    command=transcribe_audio,
    height=55,
    width=320,
    font=("Arial", 18, "bold")
)
button.pack(pady=25)

# INSTRUKSIES
guide_text = "Instructions:\n1. Choose AI Model  →  2. Choose Language  →  3. Choose Mode  →  4. Select Audio  →  5. Result saves to Downloads"
guide_label = ctk.CTkLabel(app, text=guide_text, font=("Arial", 12), text_color="gray")
guide_label.pack(side="bottom", pady=30)

app.mainloop()