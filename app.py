# app.py - SoundWave Visualizer by Loop507

import streamlit as st
import numpy as np
import cv2
import librosa
import os
import subprocess
import gc
import shutil
import contextlib
import tempfile
from typing import Tuple, Optional

# Costanti
MAX_DURATION = 300
MIN_DURATION = 1.0
MAX_FILE_SIZE = 200 * 1024 * 1024

FORMAT_RESOLUTIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:3": (800, 600)
}

VISUALIZATION_MODES = {
    "Classic Waveform": "Forma d'onda classica verticale",
    "Dense Matrix": "Matrice densa tipo griglia",
    "Frequency Spectrum": "Spettro a frequenza variabile"
}

FREQUENCY_COLOR_PRESETS = {
    "RGB Classic": {"high": "#FFFF00", "mid": "#00FF00", "low": "#FF0000"},
    "Blue Ocean": {"high": "#00FFFF", "mid": "#0080FF", "low": "#0040FF"},
    "Sunset": {"high": "#FF6600", "mid": "#FF3300", "low": "#CC0000"},
    "Neon": {"high": "#FF00FF", "mid": "#00FFFF", "low": "#FFFF00"},
    "Custom": {"high": "#FFFFFF", "mid": "#808080", "low": "#404040"}
}

def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def validate_audio_file(uploaded_file) -> bool:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File troppo grande.")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        if len(y) == 0:
            st.error("File audio vuoto.")
            return None, None, None
        audio_duration = librosa.get_duration(y=y, sr=sr)
        if audio_duration < MIN_DURATION:
            st.error("Audio troppo corto.")
            return None, None, None
        if audio_duration > MAX_DURATION:
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"Errore audio: {e}")
        return None, None, None

def generate_audio_features(y: np.ndarray, sr: int, fps: int) -> dict:
    try:
        duration = len(y) / sr
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-9)

        stft = librosa.stft(y, hop_length=512, n_fft=2048)
        magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        stft_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-9)

        n_freqs = stft_norm.shape[0]
        freq_low = stft_norm[:n_freqs//3, :]
        freq_mid = stft_norm[n_freqs//3:2*n_freqs//3, :]
        freq_high = stft_norm[2*n_freqs//3:, :]

        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)

        return {
            'mel_spectrogram': mel_norm,
            'stft_magnitude': stft_norm,
            'freq_low': freq_low,
            'freq_mid': freq_mid,
            'freq_high': freq_high,
            'rms_energy': rms_norm,
            'beats': beats,
            'tempo': tempo,
            'hop_length': 512,
            'sr': sr,
            'duration': duration
        }
    except Exception as e:
        st.error(f"Errore feature: {e}")
        return None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

def cleanup_files(*files):
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def main():
    st.set_page_config(page_title="SoundWave Visualizer", layout="centered")
    st.title("\U0001F3B5 SoundWave Visualizer")

    if not check_ffmpeg():
        st.error("FFmpeg non trovato.")
        return

    uploaded = st.file_uploader("Carica file audio", type=["wav", "mp3"])

    if uploaded:
        if not validate_audio_file(uploaded):
            return

        temp_audio = f"temp_audio_{uploaded.name}"
        with open(temp_audio, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Elaborazione audio..."):
            y, sr, duration = load_and_process_audio(temp_audio)

        if y is None:
            return

        with st.spinner("Analisi feature..."):
            features = generate_audio_features(y, sr, fps=30)

        if features is None:
            return

        st.success(f"Audio OK: {duration:.1f}s | BPM: {features['tempo']:.1f}")

        cleanup_files(temp_audio)
        gc.collect()

        st.info("Modulo generazione video in arrivo…")

if __name__ == "__main__":
    main()
