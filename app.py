import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
import tempfile
from PIL import Image, ImageDraw
from typing import Tuple, Optional, Dict, Any
import math
import random

# Costanti
MAX_DURATION = 1800  # 30 minuti
MIN_DURATION = 1.0
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
FPS_OPTIONS = [5, 10, 20, 30]

# Funzione per verificare FFmpeg
def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

# Funzione per validare il file audio
def validate_audio_file(uploaded_file) -> bool:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File troppo grande. Massimo consentito: {MAX_FILE_SIZE // (1024*1024)} MB")
        return False
    return True

# Funzione per caricare e processare l'audio
@st.cache_data
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

# Funzione per generare caratteristiche audio avanzate
@st.cache_data
def generate_enhanced_audio_features(y: np.ndarray, sr: int, fps: int) -> Optional[Dict[str, Any]]:
    try:
        duration = len(y) / sr
        hop_length = 512
        n_fft = 2048

        stft = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        stft_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-9)

        return {
            'stft_magnitude': stft_norm,
            'sr': sr,
            'duration': duration,
        }
    except Exception as e:
        st.error(f"Errore feature avanzate: {e}")
        return None

# Funzione principale
def main():
    st.set_page_config(
        page_title="SoundWave Visualizer",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 SoundWave Visualizer")
    st.markdown("*Trasforma la tua musica in arte visiva*")

    if not check_ffmpeg():
        st.error("❌ FFmpeg non trovato. Installare FFmpeg per continuare.")
        st.stop()

    uploaded_file = st.file_uploader("🎵 Carica il tuo file audio", type=['mp3', 'wav', 'flac', 'm4a', 'aac'])

    if uploaded_file:
        if not validate_audio_file(uploaded_file):
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name

        try:
            with st.spinner("🎵 Analizzando audio..."):
                y, sr, duration = load_and_process_audio(temp_audio_path)

            if y is None:
                st.error("Impossibile caricare il file audio.")
                st.stop()

            with st.spinner("🧠 Generando features audio avanzate..."):
                features = generate_enhanced_audio_features(y, sr, 30)

            if features is None:
                st.error("Errore nell'analisi audio.")
                st.stop()

            st.write(f"Sample Rate: {sr}")
            st.write(f"Duration: {duration:.2f} seconds")

            del y
            gc.collect()

        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

if __name__ == "__main__":
    main()
