# 🎙️ Whisper Transcriber — Desktop App

A simple desktop app that lets you transcribe or translate audio using OpenAI Whisper — entirely on your own computer, with no internet needed after the first setup. Just open the app, click a button, speak (or pick a file), and get your text. No coding required, ever.

---

## 📦 Model Size Guide

| Model | Size | Speed (CPU) | Accuracy | Recommendation |
|-------|------|-------------|----------|----------------|
| tiny | 75 MB | Very fast | Basic | Testing only |
| base | 145 MB | Fast | Decent | Quick notes |
| small | 466 MB | Good | Great | ⭐ Start here |
| medium | 1.5 GB | Slow | Excellent | If small isn't enough |
| large | 2.9 GB | Very slow | Best | Not recommended on CPU |

---

## 🛠️ One-Time Setup

You only need to do this once.

1. Open **Anaconda Prompt** from the Start Menu.
2. Navigate to the app folder:
   ```
   cd C:\Users\twder\WhisperT\marleyyyocode\whisper-gui
   ```
3. Double-click **`setup_env.bat`** (or run it from Anaconda Prompt).
   - This creates a Python environment and installs all dependencies.
   - On first run it will also download the Whisper model (~466 MB for `small`).
   - Wait until you see: `Setup complete! Double-click run.bat to launch the app.`

---

## ▶️ How to Use Every Time

Double-click **`run.bat`** — that's it. The app window will open.

---

## 🪟 Using the App

| Feature | How to use |
|---------|-----------|
| **Record from microphone** | Click **🎙️ Start Recording**, speak, then click **⏹ Stop Recording** |
| **Transcribe a file** | Click **📁 Open Audio File** and pick your audio file |
| **Change language** | Use the **Language** dropdown (Auto-detect works great) |
| **Translate to English** | Tick the **Translate to English** checkbox |
| **Copy text** | Click **📋 Copy to Clipboard** |
| **Save to file** | Click **💾 Save** — files go into the `transcriptions/` folder |

---

## ⚙️ How to Change the Model

1. Open `config.py` in Notepad.
2. Change the line:
   ```python
   MODEL_SIZE = "small"
   ```
   to whichever size you want (e.g. `"medium"`).
3. Save and relaunch the app. The new model will download automatically on first use.

---

## 📁 Where Files Are Saved

All saved transcriptions go into:

```
whisper-gui/transcriptions/
```

Each file is named `transcription_YYYYMMDD_HHMMSS.txt` and includes the detected language, model used, and full transcription text.

---

## 🎵 Supported Audio Formats

`.mp3` · `.wav` · `.m4a` · `.flac` · `.ogg` · `.webm`

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| "Model failed to load" | Make sure you have internet on the first run so the model can download |
| "Microphone not found" | Check Windows Sound Settings → make sure a microphone is connected and enabled |
| "Could not read audio file" | Try converting the file to `.wav` using a free tool like [Audacity](https://www.audacityteam.org/) |
| App won't start | Re-run `setup_env.bat` to ensure all dependencies are installed |
