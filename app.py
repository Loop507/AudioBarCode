# app.py - SoundWave Visualizer by Loop507 (Fixed)
import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
import tempfile
import cv2
from typing import Tuple, Optional, Dict, Any

# Costanti
MAX_DURATION: float = 300
MIN_DURATION: float = 1.0
MAX_FILE_SIZE: int = 200 * 1024 * 1024

FORMAT_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:3": (800, 600)
}

VISUALIZATION_MODES: Dict[str, str] = {
    "Classic Waveform": "Forma d'onda classica verticale",
    "Dense Matrix": "Matrice densa tipo griglia",
    "Frequency Spectrum": "Spettro a frequenza variabile"
}

FREQUENCY_COLOR_PRESETS: Dict[str, Dict[str, str]] = {
    "RGB Classic": {"high": "#FFFF00", "mid": "#00FF00", "low": "#FF0000"},
    "Blue Ocean": {"high": "#00FFFF", "mid": "#0080FF", "low": "#0040FF"},
    "Sunset": {"high": "#FF6600", "mid": "#FF3300", "low": "#CC0000"},
    "Neon": {"high": "#FF00FF", "mid": "#00FFFF", "low": "#FFFF00"},
    "Custom": {"high": "#FFFFFF", "mid": "#808080", "low": "#404040"}
}

def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def validate_audio_file(uploaded_file: st.UploadedFile) -> bool:
    """Validate the uploaded audio file."""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File troppo grande.")
        return False
    return True

@st.cache_data
def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """Load and process the audio file."""
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

@st.cache_data
def generate_audio_features(y: np.ndarray, sr: int, fps: int) -> Optional[Dict[str, Any]]:
    """Generate audio features from the audio data."""
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
    """Convert hex color to BGR format."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

def cleanup_files(*files: str) -> None:
    """Clean up temporary files."""
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception:
            pass

def generate_visualization_frame(features: Dict[str, Any], frame_idx: int, mode: str, 
                               colors: Dict[str, str], resolution: Tuple[int, int]) -> np.ndarray:
    """Generate a visualization frame based on audio features."""
    width, height = resolution
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get time index for this frame
    fps = 30
    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['mel_spectrogram'].shape[1] - 1)
    
    if mode == "Classic Waveform":
        # Simple waveform visualization
        mel_slice = features['mel_spectrogram'][:, time_idx]
        for i, intensity in enumerate(mel_slice):
            bar_height = int(intensity * height * 0.8)
            bar_x = int((i / len(mel_slice)) * width)
            color = hex_to_bgr(colors['mid'])
            cv2.rectangle(frame, (bar_x, height - bar_height), 
                         (bar_x + 3, height), color, -1)
    
    elif mode == "Dense Matrix":
        # Grid-like visualization
        cell_size = 8
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                mel_idx = min(int((y / height) * 128), 127)
                intensity = features['mel_spectrogram'][mel_idx, time_idx]
                color_val = int(intensity * 255)
                color = (color_val, color_val // 2, color_val // 4)
                cv2.rectangle(frame, (x, y), (x + cell_size, y + cell_size), color, -1)
    
    elif mode == "Frequency Spectrum":
        # Frequency-based coloring
        low_energy = np.mean(features['freq_low'][:, time_idx])
        mid_energy = np.mean(features['freq_mid'][:, time_idx])
        high_energy = np.mean(features['freq_high'][:, time_idx])
        
        # Create circles based on frequency content
        center = (width // 2, height // 2)
        low_radius = int(low_energy * min(width, height) * 0.3)
        mid_radius = int(mid_energy * min(width, height) * 0.2)
        high_radius = int(high_energy * min(width, height) * 0.1)
        
        cv2.circle(frame, center, low_radius, hex_to_bgr(colors['low']), -1)
        cv2.circle(frame, center, mid_radius, hex_to_bgr(colors['mid']), -1)
        cv2.circle(frame, center, high_radius, hex_to_bgr(colors['high']), -1)
    
    return frame

def create_video_with_opencv(frames: list, audio_path: str, fps: int, output_path: str) -> None:
    """Create a video using OpenCV and combine with audio using FFmpeg."""
    try:
        # Create temporary video without audio
        temp_video = "temp_video.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Combine video with audio using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Cleanup temp video
        cleanup_files(temp_video)
        
    except Exception as e:
        st.error(f"Errore generazione video: {e}")

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="SoundWave Visualizer", layout="centered")
    st.title("🎵 SoundWave Visualizer")
    
    if not check_ffmpeg():
        st.error("FFmpeg non trovato. Installalo per generare video.")
        return
    
    # Interface controls
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.selectbox("Modalità visualizzazione", list(VISUALIZATION_MODES.keys()))
        st.caption(VISUALIZATION_MODES[mode])
    
    with col2:
        color_preset = st.selectbox("Schema colori", list(FREQUENCY_COLOR_PRESETS.keys()))
        colors = FREQUENCY_COLOR_PRESETS[color_preset]
    
    resolution_format = st.selectbox("Formato", list(FORMAT_RESOLUTIONS.keys()))
    resolution = FORMAT_RESOLUTIONS[resolution_format]
    
    uploaded = st.file_uploader("Carica file audio", type=["wav", "mp3", "m4a", "flac"])
    
    if uploaded:
        if not validate_audio_file(uploaded):
            return
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded.read())
            temp_audio = temp_file.name
        
        try:
            with st.spinner("Elaborazione audio..."):
                y, sr, duration = load_and_process_audio(temp_audio)
            
            if y is None:
                return
            
            with st.spinner("Analisi feature audio..."):
                features = generate_audio_features(y, sr, fps=30)
            
            if features is None:
                return
            
            st.success(f"✅ Audio processato: {duration:.1f}s | BPM: {features['tempo']:.1f}")
            st.markdown("---")
            
            if st.button("🎬 Genera Visualizzazione Video"):
                fps = 30
                total_frames = int(duration * fps)
                
                with st.spinner(f"Generazione {total_frames} frame..."):
                    frames = []
                    progress_bar = st.progress(0)
                    
                    for i in range(total_frames):
                        frame = generate_visualization_frame(features, i, mode, colors, resolution)
                        frames.append(frame)
                        progress_bar.progress((i + 1) / total_frames)
                
                output_path = "output_visualization.mp4"
                
                with st.spinner("Creazione video finale..."):
                    create_video_with_opencv(frames, temp_audio, fps, output_path)
                
                if os.path.exists(output_path):
                    st.success("🎉 Video generato con successo!")
                    
                    # Show video
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "📥 Scarica Video",
                            f.read(),
                            file_name=f"soundwave_{mode.lower().replace(' ', '_')}.mp4",
                            mime="video/mp4"
                        )
                    
                    # Cleanup
                    cleanup_files(output_path)
                
        finally:
            cleanup_files(temp_audio)
            gc.collect()

if __name__ == "__main__":
    main()
