import os
import sys
import json
import threading
import webbrowser
from pathlib import Path
import customtkinter as ctk
from PIL import Image
import whisper
from tkinter import filedialog, messagebox
import anthropic
from multiprocessing import Pool
import numpy as np

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def resource_path(relative_path):
    """Get path to files inside the EXE."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

CONFIG_PATH = os.path.join(Path.home(), ".danscribe_config.json")

def load_config():
    defaults = {"api_key": "", "last_model": "Small (Accurate - 244MB)", "last_language": "Auto-Detect"}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        # Migrate old Afrikaans model names to English
        old_to_new = {
            "Base (Vinnig - 74MB)": "Base (Fast - 74MB)",
            "Small (Akkuraat - 244MB)": "Small (Accurate - 244MB)",
            "Medium (Professioneel - 769MB)": "Medium (Professional - 769MB)"
        }
        if config.get("last_model") in old_to_new:
            config["last_model"] = old_to_new[config["last_model"]]
        return config
    return defaults

def save_config(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f)

# ─────────────────────────────────────────────
#  CONFIGURATION DATA
# ─────────────────────────────────────────────

LANG_CODES = {
    "Auto-Detect": None, "Afrikaans": "af", "English": "en", "Sesotho": "st",
    "isiZulu": "zu", "isiXhosa": "xh", "Setswana": "tn", "Sepedi": "nso",
    "Siswati": "ss", "Tshivenda": "ve", "Xitsonga": "ts", "isiNdebele": "nr"
}

MODELS = {
    "Base (Fast - 74MB)": "base",
    "Small (Accurate - 244MB)": "small",
    "Medium (Professional - 769MB)": "medium"
}

# Model cache with LRU eviction — prevents memory bloat
_model_cache = {}
_MAX_CACHED_MODELS = 2

def get_model(model_name):
    """Load model with LRU caching to prevent memory bloat."""
    global _model_cache
    
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    # If cache is full, remove oldest model
    if len(_model_cache) >= _MAX_CACHED_MODELS:
        oldest_key = next(iter(_model_cache))
        del _model_cache[oldest_key]
    
    _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]

# Import docx modules at module level for performance
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime

# Load flag image once at startup to avoid repeated decoding
FLAG_IMAGE = None

def load_flag_image():
    """Load and cache the South African flag image."""
    global FLAG_IMAGE
    try:
        import io, base64
        FLAG_B64 = "/9j/4AAQSkZJRgABAQEAlgCWAAD/4QBsRXhpZgAASUkqAAgAAAADADEBAgAHAAAAMgAAABICAwACAAAAAQABAGmHBAABAAAAOgAAAAAAAABHb29nbGUAAAMAAJAHAAQAAAAwMjIwAqAEAAEAAAD6BQAAA6AEAAEAAAB5AwAAAAA="
        flag_bytes = base64.b64decode(FLAG_B64)
        flag_pil = Image.open(io.BytesIO(flag_bytes))
        FLAG_IMAGE = ctk.CTkImage(light_image=flag_pil, dark_image=flag_pil, size=(150, 90))
    except Exception:
        FLAG_IMAGE = None

# ─────────────────────────────────────────────
#  AUDIO CHUNK PROCESSING
# ─────────────────────────────────────────────

def extract_chunk_features(args):
    """Extract pitch and MFCC features from an audio chunk. For parallel processing."""
    i, seg, audio_path, sr, chunk_start_sample, chunk_end_sample = args
    try:
        import librosa
        
        # Load only this chunk instead of full audio
        y_chunk, _ = librosa.load(
            audio_path,
            sr=sr,
            mono=True,
            offset=chunk_start_sample / sr,
            duration=(chunk_end_sample - chunk_start_sample) / sr
        )
        
        if len(y_chunk) < sr * 0.3:
            return i, None
        
        # Extract pitch using YIN
        f0 = librosa.yin(y_chunk, fmin=60, fmax=400, sr=sr)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) == 0:
            mean_pitch = 0.0
            std_pitch = 0.0
        else:
            mean_pitch = float(np.mean(f0_voiced))
            std_pitch = float(np.std(f0_voiced))
        
        # Extract MFCCs for timbre
        mfcc = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=5)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        feature_vec = [mean_pitch, std_pitch] + list(mfcc_mean)
        return i, feature_vec
    except Exception as e:
        print(f"Error processing chunk {i}: {e}")
        return i, None

# ─────────────────────────────────────────────
#  SETTINGS WINDOW
# ─────────────────────────────────────────────

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("DanScribe AI — Settings")
        self.geometry("520x320")
        self.resizable(False, False)
        self.grab_set()

        config = load_config()

        ctk.CTkLabel(self, text="⚙️ Settings", font=("Arial", 20, "bold")).pack(pady=20)

        ctk.CTkLabel(self, text="Claude API Key:", font=("Arial", 13)).pack(pady=(10, 2))
        self.api_entry = ctk.CTkEntry(self, width=420, show="•", placeholder_text="sk-ant-...")
        self.api_entry.pack(pady=5)
        if config.get("api_key"):
            self.api_entry.insert(0, config["api_key"])

        # Clickable link
        link_label = ctk.CTkLabel(
            self,
            text="🔗 Get your API key at console.anthropic.com",
            font=("Arial", 11),
            text_color="#4da6ff",
            cursor="hand2"
        )
        link_label.pack(pady=4)
        link_label.bind("<Button-1>", lambda e: webbrowser.open("https://console.anthropic.com"))

        ctk.CTkLabel(
            self,
            text="Your key is stored locally on your device only.",
            font=("Arial", 10),
            text_color="gray"
        ).pack(pady=2)

        ctk.CTkButton(
            self, text="Save Settings", command=self.save,
            width=200, fg_color="#1f538d"
        ).pack(pady=20)

    def save(self):
        config = load_config()
        config["api_key"] = self.api_entry.get().strip()
        save_config(config)
        messagebox.showinfo("DanScribe AI", "Settings saved!")
        self.destroy()

# ─────────────────────────────────────────────
#  SPEAKER NAME ASSIGNMENT WINDOW
# ─────────────────────────────────────────────

class NameAssignWindow(ctk.CTkToplevel):
    def __init__(self, parent, speakers: list, callback):
        super().__init__(parent)
        self.title("Assign Speaker Names")
        self.geometry("450x420")
        self.resizable(False, False)
        self.grab_set()
        self.callback = callback
        self.entries = {}

        ctk.CTkLabel(self, text="🎤 Assign Names to Speakers", font=("Arial", 18, "bold")).pack(pady=20)
        ctk.CTkLabel(
            self,
            text="Leave blank to keep 'Speaker 1', 'Speaker 2', etc.",
            font=("Arial", 11),
            text_color="gray"
        ).pack(pady=(0, 15))

        frame = ctk.CTkScrollableFrame(self, width=380, height=220)
        frame.pack(pady=5, padx=20)

        for speaker in speakers:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=f"{speaker}:", width=110, anchor="w", font=("Arial", 12, "bold")).pack(side="left", padx=5)
            entry = ctk.CTkEntry(row, width=230, placeholder_text="Enter name...")
            entry.pack(side="left")
            self.entries[speaker] = entry

        ctk.CTkButton(
            self, text="✅ Confirm & Continue",
            command=self.confirm, width=250, fg_color="#1f538d", height=45
        ).pack(pady=20)

    def confirm(self):
        name_map = {}
        for speaker, entry in self.entries.items():
            name = entry.get().strip()
            name_map[speaker] = name if name else speaker
        self.callback(name_map)
        self.destroy()

# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class DanScribeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DanScribe AI — Professional v2.0")
        self.geometry("620x920")
        self.resizable(False, False)

        self.config_data = load_config()
        self.current_transcript = None
        self.diarized_segments = None
        self.speaker_name_map = {}

        self._build_ui()

    # ── BUILD UI ────────────────────────────

    def _build_ui(self):
        ctk.set_appearance_mode("dark")

        # ── Top banner: Logo + SA Flag side by side ──
        banner = ctk.CTkFrame(self, fg_color="transparent")
        banner.pack(pady=(10, 0), fill="x", padx=20)

        # Logo (left)
        try:
            img = Image.open(resource_path("logo.jpg"))
            logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(130, 130))
            ctk.CTkLabel(banner, image=logo_img, text="").pack(side="left", padx=(10, 0))
        except Exception:
            ctk.CTkLabel(banner, text="DanScribe AI", font=("Arial", 24, "bold")).pack(side="left", padx=10)

        # Spacer
        ctk.CTkLabel(banner, text="", fg_color="transparent").pack(side="left", expand=True)

        # SA Flag (right) — pre-loaded
        if FLAG_IMAGE:
            flag_label = ctk.CTkLabel(banner, image=FLAG_IMAGE, text="")
            flag_label.pack(side="right", padx=(0, 10))

        # Made in SA text under banner
        ctk.CTkLabel(
            self,
            text="Made in South Africa  ·  for South Africans",
            font=("Arial", 10),
            text_color="#888888"
        ).pack(pady=(2, 6))

        # Settings button (top right)
        settings_btn = ctk.CTkButton(
            self, text="⚙️ Settings", command=self.open_settings,
            width=120, height=30, fg_color="gray30"
        )
        settings_btn.place(x=480, y=10)

        # 1. Task
        ctk.CTkLabel(self, text="1. Select Task:", font=("Arial", 13, "bold")).pack(pady=(10, 2))
        self.task_var = ctk.StringVar(value="Translate to English")
        ctk.CTkOptionMenu(
            self, values=["Original Language", "Translate to English"],
            variable=self.task_var, width=380
        ).pack(pady=5)

        # 2. Language
        ctk.CTkLabel(self, text="2. Audio Language:", font=("Arial", 13, "bold")).pack(pady=(10, 2))
        self.lang_var = ctk.StringVar(value=self.config_data.get("last_language", "Auto-Detect"))
        ctk.CTkOptionMenu(
            self, values=list(LANG_CODES.keys()),
            variable=self.lang_var, width=380
        ).pack(pady=5)

        # 3. Model
        ctk.CTkLabel(self, text="3. AI Model:", font=("Arial", 13, "bold")).pack(pady=(10, 2))
        self.model_var = ctk.StringVar(value=self.config_data.get("last_model", "Small (Accurate - 244MB)"))
        ctk.CTkOptionMenu(
            self, values=list(MODELS.keys()),
            variable=self.model_var, width=380
        ).pack(pady=5)

        # 4. Speaker identification
        ctk.CTkLabel(self, text="4. Speaker Identification:", font=("Arial", 13, "bold")).pack(pady=(10, 2))
        self.diarize_var = ctk.BooleanVar(value=False)

        diarize_frame = ctk.CTkFrame(self, fg_color="transparent")
        diarize_frame.pack(pady=2)
        ctk.CTkSwitch(
            diarize_frame, text="Identify different speakers",
            variable=self.diarize_var
        ).pack(side="left", padx=10)

        # Number of speakers selector
        ctk.CTkLabel(diarize_frame, text="  Expected speakers:", font=("Arial", 11)).pack(side="left", padx=(20, 5))
        self.num_speakers_var = ctk.StringVar(value="2")
        ctk.CTkOptionMenu(
            diarize_frame,
            values=["2", "3", "4", "5", "6", "7", "8"],
            variable=self.num_speakers_var,
            width=70
        ).pack(side="left")

        # Status & progress
        self.status_label = ctk.CTkLabel(self, text="Ready", font=("Arial", 13), text_color="gray")
        self.status_label.pack(pady=(20, 5))
        self.progress_bar = ctk.CTkProgressBar(self, width=480)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)

        # Main button
        self.main_btn = ctk.CTkButton(
            self, text="▶  START PROCESSING",
            command=self.start_transcription,
            height=60, width=340,
            font=("Arial", 18, "bold"),
            fg_color="#1f538d"
        )
        self.main_btn.pack(pady=20)

        # Action buttons (after transcription)
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=5)

        self.name_btn = ctk.CTkButton(
            btn_frame, text="🎤 Assign Speaker Names",
            command=self.open_name_assignment,
            width=200, height=40, fg_color="gray30",
            state="disabled"
        )
        self.name_btn.grid(row=0, column=0, padx=10)

        self.summarize_btn = ctk.CTkButton(
            btn_frame, text="🤖 Summarize with Claude",
            command=self.summarize_with_claude,
            width=210, height=40, fg_color="#2d6a4f",
            state="disabled"
        )
        self.summarize_btn.grid(row=0, column=1, padx=10)

        # Output area
        ctk.CTkLabel(self, text="Output:", font=("Arial", 12, "bold")).pack(pady=(15, 2))
        self.output_box = ctk.CTkTextbox(self, width=560, height=180, font=("Arial", 12))
        self.output_box.pack(pady=5)

    # ── SETTINGS ────────────────────────────

    def open_settings(self):
        SettingsWindow(self)

    # ── TRANSCRIPTION ───────────────────────

    def start_transcription(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac *.mp4 *.ogg *.webm")]
        )
        if not file_path:
            return

        # Save last used settings
        self.config_data["last_model"] = self.model_var.get()
        self.config_data["last_language"] = self.lang_var.get()
        save_config(self.config_data)

        task_choice = self.task_var.get()
        lang_code = LANG_CODES[self.lang_var.get()]
        model_name = MODELS[self.model_var.get()]
        whisper_task = "translate" if task_choice == "Translate to English" else "transcribe"
        do_diarize = self.diarize_var.get()
        num_speakers = int(self.num_speakers_var.get())

        self.status_label.configure(text=f"Status: Loading model ({model_name})...")
        self.progress_bar.set(0.1)
        self.main_btn.configure(state="disabled")
        self.name_btn.configure(state="disabled")
        self.summarize_btn.configure(state="disabled")
        self.output_box.delete("1.0", "end")

        def run():
            try:
                model = get_model(model_name)
                self.progress_bar.set(0.3)
                self.status_label.configure(text="Status: Processing audio...")

                af_prompt = "Hierdie is 'n Afrikaanse transkripsie. Gebruik korrekte spelling en sinsbou."
                if lang_code == "af":
                    result = model.transcribe(file_path, task=whisper_task, language="af", initial_prompt=af_prompt)
                elif lang_code:
                    result = model.transcribe(file_path, task=whisper_task, language=lang_code)
                else:
                    result = model.transcribe(file_path, task=whisper_task)

                self.progress_bar.set(0.7)

                if do_diarize:
                    self.status_label.configure(text="Status: Identifying speakers...")
                    segments = self._diarize(result, max_speakers=num_speakers, audio_path=file_path)
                    self.diarized_segments = segments
                    transcript_text = self._segments_to_text(segments, self.speaker_name_map)
                else:
                    self.diarized_segments = None
                    transcript_text = result["text"]

                self.current_transcript = transcript_text
                output_path = self._save_transcript(transcript_text, file_path)

                self.progress_bar.set(1.0)
                self.status_label.configure(text="✅ Done!")
                self.output_box.insert("1.0", transcript_text[:2000] + ("..." if len(transcript_text) > 2000 else ""))

                if do_diarize and self.diarized_segments:
                    self.name_btn.configure(state="normal")
                self.summarize_btn.configure(state="normal")

                messagebox.showinfo("DanScribe AI", f"Transcription complete!\nSaved to folder:\n{output_path}\n\nBoth .txt and .docx files created.")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
                self.status_label.configure(text="Status: Error")

            self.main_btn.configure(state="normal")
            self.progress_bar.set(0)

        threading.Thread(target=run, daemon=True).start()

    # ── DIARIZATION ─────────────────────────

    def _diarize(self, whisper_result, max_speakers=4, audio_path=None):
        """
        Enhanced speaker detection using pitch (F0) analysis with parallel processing.
        Loads audio in chunks instead of full file to save memory.
        """
        segments = whisper_result.get("segments", [])
        if not segments:
            return [{"speaker": "Speaker 1", "text": whisper_result["text"]}]

        # ── Try pitch-based clustering with chunk loading ───────────────────
        try:
            import librosa
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            if audio_path is None:
                raise ValueError("No audio path provided")

            sr = 16000

            # Prepare tasks for parallel processing (load chunks separately)
            tasks = []
            for i, seg in enumerate(segments):
                text = seg.get("text", "").strip()
                if not text:
                    continue

                start_sample = int(seg.get("start", 0) * sr)
                end_sample = int(seg.get("end", 0) * sr)
                tasks.append((i, seg, audio_path, sr, start_sample, end_sample))

            # Process chunks in parallel
            features_list = []
            valid_indices = []
            with Pool(processes=4) as pool:
                results = pool.map(extract_chunk_features, tasks)
                for idx, feature_vec in results:
                    if feature_vec is not None:
                        features_list.append(feature_vec)
                        valid_indices.append(idx)

            if len(features_list) < max_speakers:
                raise ValueError("Not enough segments for clustering")

            # Cluster with reduced n_init for faster processing
            X = StandardScaler().fit_transform(features_list)
            km = KMeans(n_clusters=max_speakers, random_state=42, n_init=5)
            labels = km.fit_predict(X)

            # Map cluster labels to speaker numbers ordered by first appearance
            cluster_to_speaker = {}
            speaker_counter = 1
            ordered_labels = []
            for label in labels:
                if label not in cluster_to_speaker:
                    cluster_to_speaker[label] = speaker_counter
                    speaker_counter += 1
                ordered_labels.append(cluster_to_speaker[label])

            # Build result — map valid_indices back to segments
            label_map = {valid_indices[j]: ordered_labels[j] for j in range(len(valid_indices))}

            result = []
            text_buffer = []
            current_speaker = None

            for i, seg in enumerate(segments):
                text = seg.get("text", "").strip()
                if not text:
                    continue
                spk_num = label_map.get(i, 1)
                speaker_key = f"Speaker {spk_num}"

                if current_speaker is None:
                    current_speaker = speaker_key
                    text_buffer = [text]
                elif speaker_key == current_speaker:
                    text_buffer.append(text)
                else:
                    result.append({"speaker": current_speaker, "text": " ".join(text_buffer)})
                    current_speaker = speaker_key
                    text_buffer = [text]

            # Add final speaker
            if text_buffer:
                result.append({"speaker": current_speaker, "text": " ".join(text_buffer)})

            return result if result else [{"speaker": "Speaker 1", "text": whisper_result["text"]}]

        except Exception as e:
            print(f"Pitch-based diarization failed ({e}), falling back to pause analysis.")

        # ── Fallback: pause-based detection ─────────────
        pauses = []
        for i in range(1, len(segments)):
            gap = segments[i].get("start", 0) - segments[i - 1].get("end", 0)
            pauses.append((i, gap))

        sorted_pauses = sorted(pauses, key=lambda x: x[1], reverse=True)
        num_changes = max_speakers - 1
        change_points = set(idx for idx, gap in sorted_pauses[:num_changes] if gap >= 0.8)

        result = []
        current_speaker = 1
        text_buffer = []

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue
            if i in change_points and current_speaker < max_speakers:
                if text_buffer:
                    result.append({"speaker": f"Speaker {current_speaker}", "text": " ".join(text_buffer)})
                    text_buffer = []
                current_speaker += 1
            text_buffer.append(text)

        # Add final speaker
        if text_buffer:
            result.append({"speaker": f"Speaker {current_speaker}", "text": " ".join(text_buffer)})

        return result if result else [{"speaker": "Speaker 1", "text": whisper_result["text"]}]

    def _segments_to_text(self, segments, name_map=None):
        lines = []
        for seg in segments:
            speaker = seg["speaker"]
            name = name_map.get(speaker, speaker) if name_map else speaker
            lines.append(f"{name}: {seg['text']}")
        return "\n\n".join(lines)

    # ── NAME ASSIGNMENT ──────────────────────

    def open_name_assignment(self):
        if not self.diarized_segments:
            return
        # Preserve insertion order (no sorting that breaks numbering)
        seen = []
        for seg in self.diarized_segments:
            if seg["speaker"] not in seen:
                seen.append(seg["speaker"])

        def on_names_confirmed(name_map):
            self.speaker_name_map = name_map
            updated_text = self._segments_to_text(self.diarized_segments, name_map)
            self.current_transcript = updated_text
            self.output_box.delete("1.0", "end")
            self.output_box.insert("1.0", updated_text[:2000] + ("..." if len(updated_text) > 2000 else ""))
            self.status_label.configure(text="✅ Names assigned!")

        NameAssignWindow(self, seen, on_names_confirmed)

    # ── AI SUMMARY ──────────────────────────

    def summarize_with_claude(self):
        config = load_config()
        api_key = config.get("api_key", "").strip()

        if not api_key:
            messagebox.showwarning(
                "API Key Missing",
                "Please enter your Claude API key via ⚙️ Settings."
            )
            return

        if not self.current_transcript:
            messagebox.showwarning("No Transcript", "Please complete a transcription first.")
            return

        self.summarize_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Claude is generating summary...")
        self.progress_bar.set(0.3)

        def run():
            try:
                client = anthropic.Anthropic(api_key=api_key)

                prompt = f"""You are a professional minutes writer. The following is a transcription of a meeting or conversation.

Please provide a concise summary that includes:
1. Main points discussed
2. Decisions made (if any)
3. Action items (if any)
4. Participants (if names are available)

Transcription:
{self.current_transcript}

Write the summary in the same language as the transcription."""

                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )

                summary = message.content[0].text
                output_path = self._save_summary(summary)

                self.progress_bar.set(1.0)
                self.status_label.configure(text="✅ Summary complete!")
                self.output_box.delete("1.0", "end")
                self.output_box.insert("1.0", summary)

                messagebox.showinfo("DanScribe AI", f"Summary complete!\nSaved to folder:\n{output_path}\n\nBoth .txt and .docx files created.")

            except anthropic.AuthenticationError:
                messagebox.showerror("Invalid API Key", "Please check your Claude API key in Settings.")
                self.status_label.configure(text="Status: API error")
            except Exception as e:
                messagebox.showerror("Error", f"Claude API error:\n{str(e)}")
                self.status_label.configure(text="Status: Error")

            self.summarize_btn.configure(state="normal")
            self.progress_bar.set(0)

        threading.Thread(target=run, daemon=True).start()

    # ── FILE SAVING ──────────────────────────

    def _get_output_dir(self):
        path = str(Path.home() / "Downloads" / "DanScribe_Transcriptions")
        os.makedirs(path, exist_ok=True)
        return path

    def _make_docx(self, text, docx_path, doc_type="transcript"):
        """Convert text to a formatted .docx file using python-docx."""
        doc = Document()

        # Page margins
        for section in doc.sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)

        # Title
        title_text = "DanScribe AI — Meeting Summary" if doc_type == "summary" else "DanScribe AI — Transcript"
        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run(title_text)
        run.bold = True
        run.font.size = Pt(20)
        run.font.color.rgb = RGBColor(0x1f, 0x53, 0x8d)
        run.font.name = "Calibri"

        # Date
        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_run = date_para.add_run(datetime.now().strftime("%d %B %Y"))
        date_run.font.size = Pt(10)
        date_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        date_run.font.name = "Calibri"

        doc.add_paragraph()  # spacer

        # Parse and add content lines
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                doc.add_paragraph()
                continue

            # Speaker line: "Name: text"
            if doc_type == "transcript" and ":" in stripped:
                parts = stripped.split(":", 1)
                if len(parts[0]) <= 40:
                    p = doc.add_paragraph()
                    speaker_run = p.add_run(parts[0] + ": ")
                    speaker_run.bold = True
                    speaker_run.font.name = "Calibri"
                    speaker_run.font.size = Pt(11)
                    text_run = p.add_run(parts[1].strip())
                    text_run.font.name = "Calibri"
                    text_run.font.size = Pt(11)
                    continue

            # Bold markdown headings **text**
            if stripped.startswith("**") and stripped.endswith("**"):
                p = doc.add_paragraph()
                r = p.add_run(stripped.strip("*"))
                r.bold = True
                r.font.size = Pt(13)
                r.font.color.rgb = RGBColor(0x1f, 0x53, 0x8d)
                r.font.name = "Calibri"
                continue

            # Numbered headings like "1. **Main Points**"
            if stripped[:3].rstrip(". ").isdigit() and "**" in stripped:
                clean = stripped.replace("**", "").lstrip("0123456789. ")
                p = doc.add_paragraph()
                r = p.add_run(clean)
                r.bold = True
                r.font.size = Pt(13)
                r.font.color.rgb = RGBColor(0x1f, 0x53, 0x8d)
                r.font.name = "Calibri"
                continue

            # Bullet points
            if stripped.startswith("- ") or stripped.startswith("• "):
                bullet_text = stripped.lstrip("-• ").strip()
                p = doc.add_paragraph(style="List Bullet")
                r = p.add_run(bullet_text)
                r.font.name = "Calibri"
                r.font.size = Pt(11)
                continue

            # Regular paragraph
            p = doc.add_paragraph()
            r = p.add_run(stripped)
            r.font.name = "Calibri"
            r.font.size = Pt(11)

        doc.save(docx_path)

    def _save_transcript(self, text, source_path):
        base = os.path.splitext(os.path.basename(source_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._get_output_dir()
        txt_path = os.path.join(out_dir, f"{base}_transcript_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        docx_path = os.path.join(out_dir, f"{base}_transcript_{timestamp}.docx")
        try:
            self._make_docx(text, docx_path, "transcript")
        except Exception as e:
            print(f"Warning: Could not create .docx: {e}")
        return out_dir

    def _save_summary(self, text):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._get_output_dir()
        txt_path = os.path.join(out_dir, f"summary_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        docx_path = os.path.join(out_dir, f"summary_{timestamp}.docx")
        try:
            self._make_docx(text, docx_path, "summary")
        except Exception as e:
            print(f"Warning: Could not create .docx: {e}")
        return out_dir


# ─────────────────────────────────────────────
#  START APPLICATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    load_flag_image()
    ctk.set_appearance_mode("dark")
    app = DanScribeApp()
    app.mainloop()
