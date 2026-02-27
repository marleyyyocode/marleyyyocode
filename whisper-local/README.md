# 🎙️ Whisper Local — Offline Audio Transcription

A standalone local Python app that transcribes audio using [OpenAI Whisper](https://github.com/openai/whisper) — no internet required after the first run.

---

## 1. What This Is

This app is a local, persistent version of the `whisper_audio_to_text.ipynb` Colab notebook. You can:

- Record audio directly from your microphone and transcribe it
- Transcribe existing audio files (mp3, wav, m4a, flac, ogg, webm)
- Translate non-English audio to English
- Save every transcription as a timestamped `.txt` file

---

## 2. Model Choice Guide

| Model  | Download Size | Speed   | Accuracy       | Recommendation          |
|--------|--------------|---------|----------------|-------------------------|
| small  | ~466 MB      | Fast    | Good           | ✅ Start here           |
| medium | ~1.5 GB      | Slower  | Better         | Use for Spanish / mixed |
| large  | ~2.9 GB      | Slowest | Best           | Use for critical work   |

**Recommendation:** Start with `small`. You can change it any time in `config.py`.

The model downloads automatically the first time and is cached at `C:\Users\<you>\.cache\whisper\` — no manual download needed.

---

## 3. One-Time Setup

1. Open **Anaconda Prompt** from the Start Menu
2. Navigate to the repository folder:
   ```
   cd C:\path\to\marleyyyocode
   ```
3. Navigate into the app folder:
   ```
   cd whisper-local
   ```
4. Run the setup script:
   ```
   setup_env.bat
   ```
   This creates a `whisper-env` conda environment with Python 3.10 and installs all dependencies.

---

## 4. How to Use It Every Time

Double-click **`run.bat`** — that's it.

The app loads the Whisper model and shows a menu:

```
--- Whisper Local ---
  [1] Record from microphone
  [2] Transcribe an existing audio file
  [3] Translate audio to English
  [4] Show current settings
  [q] Quit
```

---

## 5. Changing the Model

Open `config.py` and change the `MODEL_SIZE` line:

```python
MODEL_SIZE = "medium"   # or "small", "large"
```

Save the file and restart the app. The new model will download automatically if not already cached.

---

## 6. Where Transcriptions Are Saved

All transcription files are saved to:

```
whisper-local/transcriptions/
```

Each file has a unique timestamped name, e.g.:

```
transcription_20240315_143022.txt
```

Files are **never overwritten** — each session creates a new file.

---

## 7. Supported Audio Formats

`mp3` · `wav` · `m4a` · `flac` · `ogg` · `webm`
