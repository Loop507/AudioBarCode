# app.py - SoundWave Visualizer by Loop507 (Streamlit Cloud Compatible)
import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor
import io
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

def validate_audio_file(uploaded_file) -> bool:
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
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            # Ensure tempo is a scalar value
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)
        except Exception:
            tempo = 120.0  # Default BPM
            beats = np.array([])
        
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

def cleanup_files(*files: str) -> None:
    """Clean up temporary files."""
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception:
            pass

def generate_visualization_frame_pil(features: Dict[str, Any], frame_idx: int, mode: str, 
                                   colors: Dict[str, str], resolution: Tuple[int, int]) -> Image.Image:
    """Generate a visualization frame using PIL."""
    width, height = resolution
    img = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img)
    
    # Get time index for this frame
    fps = 30
    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['mel_spectrogram'].shape[1] - 1)
    
    if mode == "Classic Waveform":
        # Simple waveform visualization
        mel_slice = features['mel_spectrogram'][:, time_idx]
        bar_width = max(1, width // len(mel_slice))
        
        for i, intensity in enumerate(mel_slice):
            bar_height = int(intensity * height * 0.8)
            bar_x = i * bar_width
            color = colors['mid']
            
            # Draw rectangle
            draw.rectangle([bar_x, height - bar_height, bar_x + bar_width, height], 
                         fill=color, outline=color)
    
    elif mode == "Dense Matrix":
        # Grid-like visualization using matplotlib approach with PIL
        cell_size = 8
        rows = height // cell_size
        cols = width // cell_size
        
        for row in range(rows):
            for col in range(cols):
                mel_idx = min(int((row / rows) * 128), 127)
                intensity = features['mel_spectrogram'][mel_idx, time_idx]
                
                # Create color based on intensity
                color_val = int(intensity * 255)
                color = f"#{color_val:02x}{(color_val//2):02x}{(color_val//4):02x}"
                
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=color)
    
    elif mode == "Frequency Spectrum":
        # Frequency-based coloring with circles
        low_energy = np.mean(features['freq_low'][:, time_idx])
        mid_energy = np.mean(features['freq_mid'][:, time_idx])
        high_energy = np.mean(features['freq_high'][:, time_idx])
        
        center_x, center_y = width // 2, height // 2
        
        # Draw concentric circles
        low_radius = int(low_energy * min(width, height) * 0.4)
        mid_radius = int(mid_energy * min(width, height) * 0.25)
        high_radius = int(high_energy * min(width, height) * 0.1)
        
        if low_radius > 0:
            draw.ellipse([center_x - low_radius, center_y - low_radius,
                         center_x + low_radius, center_y + low_radius], 
                        fill=colors['low'], outline=colors['low'])
        
        if mid_radius > 0:
            draw.ellipse([center_x - mid_radius, center_y - mid_radius,
                         center_x + mid_radius, center_y + mid_radius], 
                        fill=colors['mid'], outline=colors['mid'])
        
        if high_radius > 0:
            draw.ellipse([center_x - high_radius, center_y - high_radius,
                         center_x + high_radius, center_y + high_radius], 
                        fill=colors['high'], outline=colors['high'])
    
    return img

def create_video_from_images(images: list, audio_path: str, fps: int, output_path: str, resolution: Tuple[int, int]) -> None:
    """Create video from PIL images using FFmpeg."""
    try:
        # Save images to temporary files
        temp_dir = tempfile.mkdtemp()
        image_files = []
        
        for i, img in enumerate(images):
            img_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            img.save(img_path)
            image_files.append(img_path)
        
        # Create video with FFmpeg
        pattern_path = os.path.join(temp_dir, "frame_%06d.png")
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', pattern_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Errore FFmpeg: {result.stderr}")
            return
        
        # Cleanup temporary files
        for img_file in image_files:
            cleanup_files(img_file)
        os.rmdir(temp_dir)
        
    except Exception as e:
        st.error(f"Errore generazione video: {e}")

def create_preview_visualization(features: Dict[str, Any], mode: str, colors: Dict[str, str], 
                               resolution: Tuple[int, int]) -> None:
    """Create a preview of the visualization using matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if mode == "Classic Waveform":
        # Show mel spectrogram as waveform-like visualization
        mel_spec = features['mel_spectrogram']
        time_frames = np.linspace(0, features['duration'], mel_spec.shape[1])
        
        # Average across frequency bins for waveform effect
        waveform = np.mean(mel_spec, axis=0)
        ax.fill_between(time_frames, 0, waveform, color=colors['mid'], alpha=0.7)
        ax.set_title("Classic Waveform Preview")
        
    elif mode == "Dense Matrix":
        # Show mel spectrogram as matrix
        im = ax.imshow(features['mel_spectrogram'], aspect='auto', origin='lower', 
                      cmap='hot', extent=[0, features['duration'], 0, 128])
        ax.set_title("Dense Matrix Preview")
        plt.colorbar(im, ax=ax)
        
    elif mode == "Frequency Spectrum":
        # Show frequency bands over time
        time_frames = np.linspace(0, features['duration'], features['freq_low'].shape[1])
        low_avg = np.mean(features['freq_low'], axis=0)
        mid_avg = np.mean(features['freq_mid'], axis=0)
        high_avg = np.mean(features['freq_high'], axis=0)
        
        ax.plot(time_frames, low_avg, color=colors['low'], label='Low Freq', linewidth=2)
        ax.plot(time_frames, mid_avg, color=colors['mid'], label='Mid Freq', linewidth=2)
        ax.plot(time_frames, high_avg, color=colors['high'], label='High Freq', linewidth=2)
        ax.legend()
        ax.set_title("Frequency Spectrum Preview")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="SoundWave Visualizer", layout="wide")
    st.title("🎵 SoundWave Visualizer")
    st.caption("Crea visualizzazioni video dei tuoi file audio")
    
    # Check FFmpeg availability
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        st.warning("⚠️ FFmpeg non trovato. Solo anteprima disponibile.")
    
    # Interface controls
    st.sidebar.header("Impostazioni")
    
    mode = st.sidebar.selectbox("Modalità visualizzazione", list(VISUALIZATION_MODES.keys()))
    st.sidebar.caption(VISUALIZATION_MODES[mode])
    
    color_preset = st.sidebar.selectbox("Schema colori", list(FREQUENCY_COLOR_PRESETS.keys()))
    colors = FREQUENCY_COLOR_PRESETS[color_preset]
    
    # Show color preview
    st.sidebar.markdown("**Preview colori:**")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.markdown(f'<div style="background-color:{colors["high"]}; height:20px; border-radius:5px;"></div>', unsafe_allow_html=True)
        st.caption("High")
    with col2:
        st.markdown(f'<div style="background-color:{colors["mid"]}; height:20px; border-radius:5px;"></div>', unsafe_allow_html=True)
        st.caption("Mid")
    with col3:
        st.markdown(f'<div style="background-color:{colors["low"]}; height:20px; border-radius:5px;"></div>', unsafe_allow_html=True)
        st.caption("Low")
    
    resolution_format = st.sidebar.selectbox("Formato video", list(FORMAT_RESOLUTIONS.keys()))
    resolution = FORMAT_RESOLUTIONS[resolution_format]
    st.sidebar.caption(f"Risoluzione: {resolution[0]}x{resolution[1]}")
    
    # Main content
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
            
            # Show audio info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Durata", f"{duration:.1f}s")
            with col2:
                st.metric("BPM", f"{features['tempo']:.1f}")
            with col3:
                st.metric("Sample Rate", f"{sr} Hz")
            
            st.success("✅ Audio processato con successo!")
            
            # Show preview
            st.subheader("🎭 Anteprima Visualizzazione")
            create_preview_visualization(features, mode, colors, resolution)
            
            # Video generation
            if ffmpeg_available:
                st.subheader("🎬 Generazione Video")
                
                max_frames = min(300, int(duration * 30))  # Limit frames for demo
                actual_duration = max_frames / 30
                
                if actual_duration < duration:
                    st.info(f"Per la demo, genererò solo i primi {actual_duration:.1f} secondi")
                
                if st.button("🎬 Genera Video", type="primary"):
                    fps = 30
                    
                    with st.spinner(f"Generazione {max_frames} frame..."):
                        frames = []
                        progress_bar = st.progress(0)
                        
                        for i in range(max_frames):
                            frame = generate_visualization_frame_pil(features, i, mode, colors, resolution)
                            frames.append(frame)
                            progress_bar.progress((i + 1) / max_frames)
                    
                    output_path = "output_visualization.mp4"
                    
                    with st.spinner("Creazione video finale..."):
                        create_video_from_images(frames, temp_audio, fps, output_path, resolution)
                    
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
                    else:
                        st.error("❌ Errore nella generazione del video")
            else:
                st.info("💡 Installa FFmpeg per generare video")
                
        finally:
            cleanup_files(temp_audio)
            gc.collect()

if __name__ == "__main__":
    main()
