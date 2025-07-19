import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
from typing import Tuple, Optional
from moviepy.editor import AudioFileClip, ImageSequenceClip
import cv2

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

VISUALIZATION_MODES = [
    "Classic Waveform",
    "Dense Matrix",
    "Frequency Spectrum"
]

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
        if audio_duration > MAX_DURATION:
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"Errore audio: {e}")
        return None, None, None

def generate_audio_features(y: np.ndarray, sr: int, fps: int) -> Optional[dict]:
    try:
        duration = len(y) / sr
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
            st.warning(f"BPM non rilevato: {e}")

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

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

def generate_visual_frames(features: dict, duration: float, resolution: Tuple[int,int], fps: int, mode: str, color_preset: dict) -> list:
    width, height = resolution
    total_frames = int(duration * fps)

    # Convert hex colors to BGR tuples for OpenCV
    color_low = hex_to_bgr(color_preset["low"])
    color_mid = hex_to_bgr(color_preset["mid"])
    color_high = hex_to_bgr(color_preset["high"])

    frames = []
    stft = features['stft_magnitude']
    freq_low = features['freq_low']
    freq_mid = features['freq_mid']
    freq_high = features['freq_high']
    rms = features['rms_energy']

    n_time = stft.shape[1]
    samples_per_frame = max(1, n_time // total_frames)

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Indice tempo per i dati STFT
        idx = i * samples_per_frame
        if idx >= n_time:
            idx = n_time - 1

        if mode == "Classic Waveform":
            # Disegna barre verticali basate su RMS e frequenze
            bar_width = max(1, width // 60)
            spacing = bar_width
            base_y = height // 2

            for b in range(60):
                x = b * (bar_width + spacing) + 10
                # Prendi valori di intensità da freq_low, freq_mid, freq_high normalizzati
                low_val = freq_low[:, idx].mean()
                mid_val = freq_mid[:, idx].mean()
                high_val = freq_high[:, idx].mean()

                height_low = int(low_val * (height // 4))
                height_mid = int(mid_val * (height // 4))
                height_high = int(high_val * (height // 4))

                # Disegna barre verticali con colori per low/mid/high (stacked)
                cv2.rectangle(frame, (x, base_y), (x + bar_width, base_y - height_low), color_low, -1)
                cv2.rectangle(frame, (x, base_y - height_low), (x + bar_width, base_y - height_low - height_mid), color_mid, -1)
                cv2.rectangle(frame, (x, base_y - height_low - height_mid), (x + bar_width, base_y - height_low - height_mid - height_high), color_high, -1)

        elif mode == "Dense Matrix":
            # Crea una griglia colorata basata sul valore STFT all'indice
            grid_rows = 30
            grid_cols = 40
            cell_width = width // grid_cols
            cell_height = height // grid_rows

            for row in range(grid_rows):
                for col in range(grid_cols):
                    # Prendi un valore da stft in funzione di riga, colonna e tempo
                    freq_idx = int(row * (stft.shape[0] / grid_rows))
                    time_idx = min(idx + col, stft.shape[1] - 1)
                    val = stft[freq_idx, time_idx]
                    # Colore tra low-mid-high interpolato
                    if freq_idx < stft.shape[0] // 3:
                        color = color_low
                    elif freq_idx < 2 * stft.shape[0] // 3:
                        color = color_mid
                    else:
                        color = color_high
                    # Modula l'intensità
                    intensity = int(val * 255)
                    bgr = tuple(min(255, int(c * (intensity / 255))) for c in color)
                    top_left = (col * cell_width, row * cell_height)
                    bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
                    cv2.rectangle(frame, top_left, bottom_right, bgr, -1)

        elif mode == "Frequency Spectrum":
            # Spettro orizzontale: disegna barre per low, mid, high frequenze
            bar_count = 30
            bar_width = width // (bar_count * 2)
            spacing = bar_width
            base_y = height - 20

            for b in range(bar_count):
                x = b * (bar_width + spacing) + 10

                # Media valori frequenze intorno a b nel tempo idx
                low_idx = min(b, freq_low.shape[0]-1)
                mid_idx = min(b, freq_mid.shape[0]-1)
                high_idx = min(b, freq_high.shape[0]-1)

                low_val = freq_low[low_idx, idx]
                mid_val = freq_mid[mid_idx, idx]
                high_val = freq_high[high_idx, idx]

                height_low = int(low_val * (height // 3))
                height_mid = int(mid_val * (height // 3))
                height_high = int(high_val * (height // 3))

                # Barre da basso verso l'alto
                cv2.rectangle(frame, (x, base_y), (x + bar_width, base_y - height_low), color_low, -1)
                cv2.rectangle(frame, (x + bar_width, base_y), (x + 2 * bar_width, base_y - height_mid), color_mid, -1)
                cv2.rectangle(frame, (x + 2 * bar_width, base_y), (x + 3 * bar_width, base_y - height_high), color_high, -1)

        frames.append(frame)

    return frames

def create_video_with_audio(frames: list, audio_path: str, fps: int, output_path: str):
    try:
        clip = ImageSequenceClip(frames, fps=fps)
        audio = AudioFileClip(audio_path)
        final = clip.set_audio(audio)
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        st.error(f"Errore generazione video: {e}")

def cleanup_files(*files):
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass

def main():
    st.set_page_config(page_title="SoundWave Visualizer by Loop507", layout="centered")
    st.title("\U0001F3B5 SoundWave Visualizer ")
    st.markdown("<small>by Loop507</small>", unsafe_allow_html=True)

    if not check_ffmpeg():
        st.error("FFmpeg non trovato.")
        return

    uploaded = st.file_uploader("Carica file audio", type=["wav", "mp3"])
    if not uploaded:
        st.info("Carica un file audio WAV o MP3 (max 200MB)")
        return

    if not validate_audio_file(uploaded):
        return

    temp_audio = f"temp_audio_{uploaded.name}"
    with open(temp_audio, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Elaborazione audio..."):
        y, sr, duration = load_and_process_audio(temp_audio)

    if y is None:
        return

    fps = st.selectbox("Seleziona FPS", options=[5,10,20,30], index=3)
    mode = st.selectbox("Modalità Visualizzazione", options=VISUALIZATION_MODES, index=0)
    color_preset_name = st.selectbox("Seleziona palette colori", options=list(FREQUENCY_COLOR_PRESETS.keys()), index=0)
    color_preset = FREQUENCY_COLOR_PRESETS[color_preset_name]

    with st.spinner("Analisi feature..."):
        features = generate_audio_features(y, sr, fps)

    if features is None:
        return

    st.success(f"✅ Audio OK: {duration:.1f}s | BPM: {features['tempo']:.1f}")

    st.markdown("---")
    if st.button("\U0001F3AC Genera Video"):
        with st.spinner("Generazione video..."):
            resolution = FORMAT_RESOLUTIONS["16:9"]  # Puoi aggiungere selettore formato se vuoi
            frames = generate_visual_frames(features, duration, resolution, fps, mode, color_preset)
            output_path = "output_video.mp4"
            create_video_with_audio(frames, temp_audio, fps, output_path)

            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button("Scarica Video", f, file_name="output_video.mp4", mime="video/mp4")
                st.video(output_path)

    cleanup_files(temp_audio)
    gc.collect()

if __name__ == "__main__":
    main()
