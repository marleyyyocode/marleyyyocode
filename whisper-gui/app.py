import os
import datetime
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
}


class WhisperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎙️ Whisper Transcriber")
        self.resizable(False, False)

        self.model = None
        self._recording = False
        self._audio_chunks = []
        self._stream = None

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self._build_ui()
        self._set_buttons_enabled(False)
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        top = tk.Frame(self)
        top.pack(fill="x", **pad)

        tk.Label(top, text="Language:").pack(side="left")
        self._lang_var = tk.StringVar(value="Auto-detect")
        lang_menu = ttk.Combobox(
            top,
            textvariable=self._lang_var,
            values=list(LANGUAGES.keys()),
            state="readonly",
            width=14,
        )
        lang_menu.pack(side="left", padx=(4, 0))

        self._translate_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self, text="Translate to English", variable=self._translate_var
        ).pack(anchor="w", padx=10)

        self._rec_btn = tk.Button(
            self,
            text="🎙️ Start Recording",
            bg="#2ecc71",
            fg="white",
            font=("", 11, "bold"),
            width=22,
            command=self._toggle_recording,
        )
        self._rec_btn.pack(**pad)

        tk.Label(self, text="— or transcribe a file —", fg="gray").pack()

        self._file_btn = tk.Button(
            self,
            text="📁 Open Audio File",
            width=22,
            command=self._open_file,
        )
        self._file_btn.pack(**pad)

        self._status_var = tk.StringVar(value="Loading model, please wait...")
        tk.Label(self, textvariable=self._status_var, anchor="w").pack(
            fill="x", padx=10
        )

        self._text_box = tk.Text(self, width=55, height=12, wrap="word")
        self._text_box.pack(padx=10, pady=5)

        btn_row = tk.Frame(self)
        btn_row.pack(pady=(0, 10))
        self._copy_btn = tk.Button(
            btn_row, text="📋 Copy to Clipboard", command=self._copy
        )
        self._copy_btn.pack(side="left", padx=5)
        self._save_btn = tk.Button(btn_row, text="💾 Save", command=self._save)
        self._save_btn.pack(side="left", padx=5)

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for w in (self._rec_btn, self._file_btn, self._copy_btn, self._save_btn):
            w.config(state=state)

    def _set_status(self, msg):
        self._status_var.set(f"Status: {msg}")

    # ------------------------------------------------------- Model loading --

    def _load_model(self):
        try:
            import whisper
            self.model = whisper.load_model(config.MODEL_SIZE)
            self.after(0, self._on_model_loaded)
        except Exception:
            self.after(
                0,
                lambda: self._set_status(
                    "❌ Model failed to load. Check your internet connection for first run."
                ),
            )

    def _on_model_loaded(self):
        self._set_status("✅ Model loaded. Ready.")
        self._set_buttons_enabled(True)

    # ------------------------------------------------------- Recording ------

    def _toggle_recording(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._audio_chunks = []
        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            self._set_status(
                "❌ Microphone not found. Check your audio settings."
            )
            return

        self._recording = True
        self._rec_btn.config(text="⏹ Stop Recording", bg="#e74c3c")
        self._set_status("Recording...")

    def _audio_callback(self, indata, frames, time, status):
        self._audio_chunks.append(indata.copy())

    def _stop_recording(self):
        self._recording = False
        self._rec_btn.config(text="🎙️ Start Recording", bg="#2ecc71")
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._audio_chunks:
            self._set_status("Ready")
            return

        audio = np.concatenate(self._audio_chunks, axis=0)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(config.OUTPUT_DIR, f"recording_{ts}.wav")
        sf.write(wav_path, audio, config.SAMPLE_RATE)

        self._set_buttons_enabled(False)
        self._set_status("Transcribing...")
        threading.Thread(
            target=self._transcribe,
            args=(wav_path, wav_path, True),
            daemon=True,
        ).start()

    # ------------------------------------------------------- File picker ----

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                (
                    "Audio files",
                    "*.mp3 *.wav *.m4a *.flac *.ogg *.webm",
                ),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._set_buttons_enabled(False)
        self._set_status("Transcribing...")
        threading.Thread(
            target=self._transcribe,
            args=(path, path, False),
            daemon=True,
        ).start()

    # ------------------------------------------------------- Transcription --

    def _transcribe(self, audio_path, source_label, is_temp):
        try:
            lang = LANGUAGES[self._lang_var.get()]
            task = "translate" if self._translate_var.get() else "transcribe"
            result = self.model.transcribe(audio_path, language=lang, task=task)
        except Exception:
            self.after(
                0,
                lambda: self._set_status(
                    "❌ Could not read audio file. Try a different format."
                ),
            )
            self.after(0, lambda: self._set_buttons_enabled(True))
            return
        finally:
            if is_temp and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        text = result.get("text", "").strip()
        lang_detected = result.get("language", "unknown")
        self.after(0, lambda: self._show_result(text, lang_detected, source_label))

    def _show_result(self, text, lang_detected, source_label):
        self._text_box.delete("1.0", tk.END)
        self._text_box.insert(tk.END, text)
        self._last_lang = lang_detected
        self._last_source = source_label
        self._set_status("Done!")
        self._set_buttons_enabled(True)

    # ------------------------------------------------------- Clipboard / Save

    def _copy(self):
        text = self._text_box.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self._set_status("Copied!")
        self.after(2000, lambda: self._set_status("Ready"))

    def _save(self):
        text = self._text_box.get("1.0", tk.END).strip()
        if not text:
            self._set_status("Nothing to save.")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{ts}.txt"
        path = os.path.join(config.OUTPUT_DIR, filename)

        lang = getattr(self, "_last_lang", "unknown")
        source = getattr(self, "_last_source", "unknown")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {ts}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Language detected: {lang}\n")
            f.write(f"Model: {config.MODEL_SIZE}\n")
            f.write("=" * 60 + "\n\n")
            f.write(text + "\n")

        self._set_status(f"Saved to {config.OUTPUT_DIR}/{filename}")


if __name__ == "__main__":
    app = WhisperApp()
    app.mainloop()
