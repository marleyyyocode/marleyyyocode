import os
import sys
import datetime
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

import config


def ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def timestamped_filename(prefix="transcription", ext=".txt"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}{ext}"
    return os.path.join(config.OUTPUT_DIR, name)


def save_result(source, result, task="transcribe"):
    path = timestamped_filename()
    detected = result.get("language", "unknown")
    label = "TRANSLATION (to English)" if task == "translate" else "TRANSCRIPTION"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Source: {source}\n")
        f.write(f"Language: {detected}\n")
        f.write(f"Model: {config.MODEL_SIZE}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"FULL {label}:\n")
        f.write(result["text"].strip() + "\n\n")
        f.write("TIMESTAMPED SEGMENTS:\n")
        for seg in result["segments"]:
            f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}\n")
    return path


def print_result(result, task="transcribe"):
    detected = result.get("language", "unknown")
    label = "TRANSLATION (to English)" if task == "translate" else "TRANSCRIPTION"
    print(f"\nDetected language: {detected}")
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(result["text"].strip())
    print("\n" + f"{'Start':>8}  {'End':>8}  Text")
    print("-" * 60)
    for seg in result["segments"]:
        print(f"{seg['start']:>7.1f}s  {seg['end']:>7.1f}s  {seg['text'].strip()}")


def record_audio(seconds):
    print(f"Recording for {seconds} second(s)... (press Ctrl+C to cancel)")
    audio = sd.rec(
        int(seconds * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio


def get_seconds(default=10):
    raw = input(f"  Seconds to record [default {default}]: ").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        print("  Invalid input, using default.")
        return default


def get_file_path():
    raw = input("  Audio file path: ").strip().strip('"').strip("'")
    return raw


def run_transcription(model, audio_path, source_label, task="transcribe"):
    lang = config.LANGUAGE if task == "transcribe" else None
    print(f"\nProcessing '{source_label}'...")
    result = model.transcribe(audio_path, language=lang, task=task)
    print_result(result, task)
    saved = save_result(source_label, result, task)
    print(f"\nSaved to: {saved}")


def option_mic(model, task="transcribe"):
    seconds = get_seconds()
    try:
        audio = record_audio(seconds)
    except sd.PortAudioError as exc:
        print(f"  Microphone error: {exc}")
        return
    except KeyboardInterrupt:
        print("  Recording cancelled.")
        return

    wav_path = timestamped_filename(prefix="recording", ext=".wav")
    sf.write(wav_path, audio, config.SAMPLE_RATE)
    print(f"  Audio saved to: {wav_path}")

    try:
        run_transcription(model, wav_path, wav_path, task)
    except Exception as exc:
        print(f"  Transcription error: {exc}")


def option_file(model, task="transcribe"):
    path = get_file_path()
    if not os.path.isfile(path):
        print(f"  File not found: {path}")
        return
    try:
        run_transcription(model, path, path, task)
    except Exception as exc:
        print(f"  Transcription error: {exc}")


def option_translate(model):
    print("\n  [1] Record from microphone")
    print("  [2] Transcribe an existing file")
    choice = input("  Choice: ").strip()
    if choice == "1":
        option_mic(model, task="translate")
    elif choice == "2":
        option_file(model, task="translate")
    else:
        print("  Invalid choice.")


def show_settings():
    print(f"\n  Model:         {config.MODEL_SIZE}")
    print(f"  Language:      {config.LANGUAGE if config.LANGUAGE else 'auto-detect'}")
    print(f"  Output folder: {os.path.abspath(config.OUTPUT_DIR)}")


def main():
    ensure_output_dir()

    print(f"Loading Whisper '{config.MODEL_SIZE}' model...")
    model = whisper.load_model(config.MODEL_SIZE)
    print(f"Model '{config.MODEL_SIZE}' loaded.\n")

    while True:
        print("\n--- Whisper Local ---")
        print("  [1] Record from microphone")
        print("  [2] Transcribe an existing audio file")
        print("  [3] Translate audio to English")
        print("  [4] Show current settings")
        print("  [q] Quit")
        choice = input("Choice: ").strip().lower()

        if choice == "1":
            option_mic(model)
        elif choice == "2":
            option_file(model)
        elif choice == "3":
            option_translate(model)
        elif choice == "4":
            show_settings()
        elif choice == "q":
            print("Goodbye.")
            sys.exit(0)
        else:
            print("  Invalid choice, try again.")


if __name__ == "__main__":
    main()
