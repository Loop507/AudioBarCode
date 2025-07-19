# app.py - SoundWave Visualizer by Loop507

import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
from typing import Tuple, Optional
from moviepy.editor import AudioFileClip, ImageSequenceClip
from PIL import Image, ImageDraw

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

def generate_audio_features(y: np.ndarray, sr: int, fps: int) -> Optional[dict]:
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

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def generate_visual_frames(features: dict, duration: float, resolution: Tuple[int,int], fps: int, mode: str, color_preset: dict) -> list:
    width, height = resolution
    total_frames = int(duration * fps)

    color_low = hex_to_rgb(color_preset["low"])
    color_mid = hex_to_rgb(color_preset["mid"])
    color_high = hex_to_rgb(color_preset["high"])

    frames = []
    stft = features['stft_magnitude']
    freq_low = features['freq_low']
    freq_mid = features['freq_mid']
    freq_high = features['freq_high']

    n_time = stft.shape[1]
    samples_per_frame = max(1, n_time // total_frames)

    for i in range(total_frames):
        frame_img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(frame_img)

        idx = i * samples_per_frame
        if idx >= n_time:
            idx = n_time - 1

        if mode == "Classic Waveform":
            bar_width = max(1, width // 60)
            spacing = bar_width
            base_y = height // 2

            for b in range(60):
                x = b * (bar_width + spacing) + 10

                low_val = freq_low[:, idx].mean()
                mid_val = freq_mid[:, idx].mean()
                high_val = freq_high[:, idx].mean()

                height_low = int(low_val * (height // 4))
                height_mid = int(mid_val * (height // 4))
                height_high = int(high_val * (height // 4))

                draw.rectangle([x, base_y - height_low, x + bar_width, base_y], fill=color_low)
                draw.rectangle([x, base_y - height_low - height_mid, x + bar_width, base_y - height_low], fill=color_mid)
                draw.rectangle([x, base_y - height_low - height_mid - height_high, x + bar_width, base_y - height_low - height_mid], fill=color_high)

        elif mode == "Dense Matrix":
            grid_rows = 30
            grid_cols = 40
            cell_width = width // grid_cols
            cell_height = height // grid_rows

            for row in range(grid_rows):
                for col in range(grid_cols):
                    freq_idx = int(row * (stft.shape[0] / grid_rows))
                    time_idx = min(idx + col, stft.shape[1] - 1)
                    val = stft[freq_idx, time_idx]

                    if freq_idx < stft.shape[0] // 3:
                        color = color_low
                    elif freq_idx < 2 * stft.shape[0] // 3:
                        color = color_mid
                    else:
                        color = color_high

                    # Modula l'intensità del colore
                    r = int(color[0] * val)
                    g = int(color[1] * val)
                    b = int(color[2] * val)
                    fill_color = (r, g, b)

                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height

                    draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        elif mode == "Frequency Spectrum":
            bar_count = 30
            bar_width = width // (bar_count * 3)
            spacing = bar_width
            base_y = height - 20

            for b in range(bar_count):
                x = b * (bar_width + spacing) + 10

                low_idx = min(b, freq_low.shape[0]-1)
                mid_idx = min(b, freq_mid.shape[0]-1)
                high_idx = min(b, freq_high.shape[0]-1)

                low_val = freq_low[low_idx, idx]
                mid_val = freq_mid[mid_idx, idx]
                high_val = freq_high[high_idx, idx]

                height_low = int(low_val * (height // 3))
                height_mid = int(mid_val * (height // 3))
                height_high = int(high_val * (height // 3))

                draw.rectangle([x, base_y - height_low, x + bar_width, base_y], fill=color_low)
                draw.rectangle([x + bar_width, base_y - height_mid, x + 2*bar_width, base_y], fill=color_mid)
                draw.rectangle([x + 2*bar_width, base_y - height_high, x + 3*bar_width, base_y], fill=color_high)

        frames.append(np.array(frame_img))

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
    st.title("\U0001F3B5 SoundWave Visualizer")
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
    mode = st.selectbox("Modalità Visualizzazione", options=list(VISUALIZATION_MODES.keys()), index=0)
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
            resolution = FORMAT_RESOLUTIONS["16:9"]  # puoi aggiungere selettore formato
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
