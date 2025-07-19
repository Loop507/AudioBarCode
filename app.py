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
MAX_DURATION = 300  # 5 minuti max
MIN_DURATION = 1.0
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB max

FORMAT_RESOLUTIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:3": (800, 600)
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
        if audio_duration > MAX_DURATION:
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"Errore caricamento audio: {e}")
        return None, None, None

def generate_audio_features(y: np.ndarray, sr: int, fps: int) -> Optional[dict]:
    try:
        duration = len(y) / sr

        # Calcolo spettrogramma STFT
        stft = librosa.stft(y, hop_length=512, n_fft=2048)
        magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        stft_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-9)

        n_freqs = stft_norm.shape[0]
        freq_low = stft_norm[:n_freqs//3, :]
        freq_mid = stft_norm[n_freqs//3:2*n_freqs//3, :]
        freq_high = stft_norm[2*n_freqs//3:, :]

        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

        tempo, beats = 0.0, []
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        except Exception as e:
            st.warning(f"Warning beat track fallito: {e}")

        return {
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
        st.error(f"Errore feature audio: {e}")
        return None

def cleanup_files(*files):
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def generate_dummy_frames(duration: float, resolution: Tuple[int, int], fps: int) -> list:
    width, height = resolution
    total_frames = int(duration * fps)
    return [np.random.randint(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(total_frames)]

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
    st.title("\U0001F3B5 SoundWave Visualizer by Loop507")

    if not check_ffmpeg():
        st.error("FFmpeg non trovato.")
        return

    uploaded = st.file_uploader("Carica file audio", type=["wav", "mp3"])

    fps = st.selectbox("Seleziona FPS", options=[5, 10, 20, 30], index=3)
    formato = st.selectbox("Seleziona formato video", options=list(FORMAT_RESOLUTIONS.keys()), index=0)

    if uploaded:
        if not validate_audio_file(uploaded):
            return

        temp_audio = f"temp_audio_{uploaded.name}"
        with open(temp_audio, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Elaborazione audio..."):
            y, sr, duration = load_and_process_audio(temp_audio)

        if y is None:
            cleanup_files(temp_audio)
            return

        with st.spinner("Analisi feature audio..."):
            features = generate_audio_features(y, sr, fps=fps)

        st.write("Features ottenute:", features)  # debug

        if not features or 'tempo' not in features or features['tempo'] is None:
            st.error("Analisi tempo non disponibile o errore nelle feature.")
            cleanup_files(temp_audio)
            return

        st.success(f"Audio OK: {duration:.1f}s | BPM: {features['tempo']:.1f}")

        st.markdown("---")

        if st.button("\U0001F3AC Genera Video Placeholder"):
            with st.spinner("Generazione video..."):
                resolution = FORMAT_RESOLUTIONS[formato]
                dummy_frames = generate_dummy_frames(duration, resolution, fps)
                output_path = "output_video.mp4"
                create_video_with_audio(dummy_frames, temp_audio, fps, output_path)
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button("Scarica Video", f, file_name="output_video.mp4", mime="video/mp4")
                    st.video(output_path)

        cleanup_files(temp_audio)
        gc.collect()

if __name__ == "__main__":
    main()
