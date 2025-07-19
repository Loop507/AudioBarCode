# app.py - SoundWave Visualizer by Loop507

import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
from typing import Tuple, Optional
from moviepy.editor import AudioFileClip, ImageSequenceClip

# Costanti
MAX_DURATION = 300  # Rimane il max 5 minuti
MIN_DURATION = 1.0
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

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
        st.error("File troppo grande (max 200 MB).")
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
        # RIMOSSO IL TAGLIO A 10 SECONDI
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
        # STFT ANALISI come nella demo
        hop_length = 512
        n_fft = 2048

        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        stft_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-9)

        n_freqs = stft_norm.shape[0]
        freq_low = stft_norm[:n_freqs//3, :]
        freq_mid = stft_norm[n_freqs//3:2*n_freqs//3, :]
        freq_high = stft_norm[2*n_freqs//3:, :]

        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

        return {
            'stft_magnitude': stft_norm,
            'freq_low': freq_low,
            'freq_mid': freq_mid,
            'freq_high': freq_high,
            'rms_energy': rms_norm,
            'beats': beats,
            'tempo': tempo,
            'hop_length': hop_length,
            'sr': sr,
            'duration': duration
        }
    except Exception as e:
        st.error(f"Errore feature: {e}")
        return None

def generate_dummy_frames(duration: float, resolution: Tuple[int, int], fps: int) -> list:
    width, height = resolution
    total_frames = int(duration * fps)
    return [(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)) for _ in range(total_frames)]

def create_video_with_audio(frames: list, audio_path: str, fps: int, output_path: str):
    try:
        clip = ImageSequenceClip(frames, fps=fps)
        audio = AudioFileClip(audio_path)
        final = clip.set_audio(audio)
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        st.error(f"Errore generazione video: {e}")

def main():
    st.set_page_config(page_title="SoundWave Visualizer by Loop507", layout="centered")
    st.markdown("# \U0001F3B5 SoundWave Visualizer  <small>by Loop507</small>", unsafe_allow_html=True)

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

        fps = st.selectbox("Seleziona FPS", options=[5, 10, 20, 30], index=3)  # Default 30 fps

        with st.spinner("Elaborazione audio..."):
            y, sr, duration = load_and_process_audio(temp_audio)

        if y is None:
            cleanup_files(temp_audio)
            return

        with st.spinner("Analisi feature..."):
            features = generate_audio_features(y, sr, fps)

        if features is None:
            cleanup_files(temp_audio)
            return

        st.success(f"Audio OK: {duration:.1f}s | BPM: {features['tempo']:.1f}")

        st.markdown("---")
        if st.button("\U0001F3AC Genera Video Placeholder"):
            with st.spinner("Generazione video..."):
                dummy_frames = generate_dummy_frames(duration, (1280, 720), fps)
                output_path = "output_video.mp4"
                create_video_with_audio(dummy_frames, temp_audio, fps, output_path)
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button("Scarica Video", f, file_name="output_video.mp4", mime="video/mp4")
                    st.video(output_path)

        cleanup_files(temp_audio)
        gc.collect()

def cleanup_files(*files):
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

if __name__ == "__main__":
    main()
