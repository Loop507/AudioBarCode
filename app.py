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

# Costanti - AGGIORNATE
MAX_DURATION: float = 1800  # 30 minuti invece di 5 minuti
MIN_DURATION: float = 1.0
MAX_FILE_SIZE: int = 200 * 1024 * 1024  # 200 MB confermato

# Opzioni FPS disponibili
FPS_OPTIONS: list = [5, 10, 20, 30]

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
        st.error(f"File troppo grande. Massimo consentito: {MAX_FILE_SIZE // (1024*1024)} MB")
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
    """Generate audio features from the audio data using STFT analysis."""
    try:
        duration = len(y) / sr
        
        # STFT Analysis - Spettro principale come richiesto
        hop_length = 512
        n_fft = 2048
        stft = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Normalizzazione per visualizzazione
        stft_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-9)
        
        # Suddivisione in bande di frequenza
        n_freqs = stft_norm.shape[0]
        freq_low = stft_norm[:n_freqs//3, :]      # Frequenze basse
        freq_mid = stft_norm[n_freqs//3:2*n_freqs//3, :]  # Frequenze medie
        freq_high = stft_norm[2*n_freqs//3:, :]   # Frequenze acute
        
        # Mel spectrogram per visualizzazioni aggiuntive
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-9)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
        
        # Rilevamento del tempo
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)
        except Exception:
            tempo = 120.0
            beats = np.array([])
        
        return {
            'stft_magnitude': stft_norm,          # Spettro STFT principale
            'freq_low': freq_low,
            'freq_mid': freq_mid,
            'freq_high': freq_high,
            'mel_spectrogram': mel_norm,          # Per compatibilità
            'rms_energy': rms_norm,
            'beats': beats,
            'tempo': tempo,
            'hop_length': hop_length,
            'sr': sr,
            'duration': duration,
            'magnitude_raw': magnitude            # Per analisi avanzate
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
                                   colors: Dict[str, str], resolution: Tuple[int, int], fps: int) -> Image.Image:
    """Generate a visualization frame using PIL with STFT data."""
    width, height = resolution
    img = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img)
    
    # Calcola indice temporale basato su FPS
    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['stft_magnitude'].shape[1] - 1)
    
    if mode == "Classic Waveform":
        # Visualizzazione waveform usando STFT
        spectrum_slice = features['stft_magnitude'][:, time_idx]
        n_bars = min(width // 3, len(spectrum_slice))
        bar_width = width // n_bars
        
        for i in range(n_bars):
            freq_idx = int(i * len(spectrum_slice) / n_bars)
            intensity = spectrum_slice[freq_idx]
            bar_height = int(intensity * height * 0.8)
            bar_x = i * bar_width
            
            # Colore basato sulla frequenza
            if i < n_bars // 3:
                color = colors['low']
            elif i < 2 * n_bars // 3:
                color = colors['mid']
            else:
                color = colors['high']
            
            if bar_height > 0:
                draw.rectangle([bar_x, height - bar_height, bar_x + bar_width - 1, height], 
                             fill=color, outline=color)
    
    elif mode == "Dense Matrix":
        # Matrice densa usando spettro STFT
        cell_size = 6
        rows = height // cell_size
        cols = width // cell_size
        
        spectrum = features['stft_magnitude'][:, time_idx]
        
        for row in range(rows):
            for col in range(cols):
                # Mappa la posizione alla frequenza
                freq_idx = min(int((row / rows) * len(spectrum)), len(spectrum) - 1)
                intensity = spectrum[freq_idx]
                
                # Effetto temporale
                time_offset = col / cols
                intensity_adjusted = intensity * (0.7 + 0.3 * np.sin(time_offset * np.pi))
                
                # Colore basato sull'intensità
                color_val = int(intensity_adjusted * 255)
                color = f"#{color_val:02x}{max(0, color_val-50):02x}{max(0, color_val-100):02x}"
                
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = x1 + cell_size - 1, y1 + cell_size - 1
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=color)
    
    elif mode == "Frequency Spectrum":
        # Visualizzazione con cerchi basata su bande di frequenza STFT
        low_energy = np.mean(features['freq_low'][:, time_idx])
        mid_energy = np.mean(features['freq_mid'][:, time_idx])
        high_energy = np.mean(features['freq_high'][:, time_idx])
        
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2 - 20
        
        # Cerchi concentrici con dimensioni basate sull'energia
        low_radius = int(low_energy * max_radius * 0.8)
        mid_radius = int(mid_energy * max_radius * 0.5)
        high_radius = int(high_energy * max_radius * 0.2)
        
        # Disegna dal più grande al più piccolo
        if low_radius > 5:
            draw.ellipse([center_x - low_radius, center_y - low_radius,
                         center_x + low_radius, center_y + low_radius], 
                        fill=colors['low'], outline=colors['low'])
        
        if mid_radius > 3:
            draw.ellipse([center_x - mid_radius, center_y - mid_radius,
                         center_x + mid_radius, center_y + mid_radius], 
                        fill=colors['mid'], outline=colors['mid'])
        
        if high_radius > 1:
            draw.ellipse([center_x - high_radius, center_y - high_radius,
                         center_x + high_radius, center_y + high_radius], 
                        fill=colors['high'], outline=colors['high'])
    
    return img

def create_video_from_images(images: list, audio_path: str, fps: int, output_path: str, resolution: Tuple[int, int]) -> None:
    """Create video from PIL images using FFmpeg."""
    try:
        # Salva le immagini in file temporanei
        temp_dir = tempfile.mkdtemp()
        image_files = []
        
        for i, img in enumerate(images):
            img_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            img.save(img_path)
            image_files.append(img_path)
        
        # Crea video con FFmpeg
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
        
        # Pulizia file temporanei
        for img_file in image_files:
            cleanup_files(img_file)
        os.rmdir(temp_dir)
        
    except Exception as e:
        st.error(f"Errore generazione video: {e}")

def create_preview_visualization(features: Dict[str, Any], mode: str, colors: Dict[str, str], 
                               resolution: Tuple[int, int]) -> None:
    """Create a preview of the visualization using STFT data."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if mode == "Classic Waveform":
        # Visualizza lo spettro STFT come waveform
        stft_spec = features['stft_magnitude']
        time_frames = np.linspace(0, features['duration'], stft_spec.shape[1])
        
        # Media lungo le frequenze per effetto waveform
        waveform = np.mean(stft_spec, axis=0)
        ax.fill_between(time_frames, 0, waveform, color=colors['mid'], alpha=0.7)
        ax.set_title("Classic Waveform Preview (STFT)")
        
    elif mode == "Dense Matrix":
        # Mostra spettrogramma STFT come matrice
        im = ax.imshow(features['stft_magnitude'], aspect='auto', origin='lower', 
                      cmap='hot', extent=[0, features['duration'], 0, features['stft_magnitude'].shape[0]])
        ax.set_title("Dense Matrix Preview (STFT)")
        plt.colorbar(im, ax=ax)
        
    elif mode == "Frequency Spectrum":
        # Mostra bande di frequenza nel tempo
        time_frames = np.linspace(0, features['duration'], features['freq_low'].shape[1])
        low_avg = np.mean(features['freq_low'], axis=0)
        mid_avg = np.mean(features['freq_mid'], axis=0)
        high_avg = np.mean(features['freq_high'], axis=0)
        
        ax.plot(time_frames, low_avg, color=colors['low'], label='Low Freq', linewidth=2)
        ax.plot(time_frames, mid_avg, color=colors['mid'], label='Mid Freq', linewidth=2)
        ax.plot(time_frames, high_avg, color=colors['high'], label='High Freq', linewidth=2)
        ax.legend()
        ax.set_title("Frequency Spectrum Preview (STFT)")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="SoundWave Visualizer by Loop507", layout="wide")
    
    # Titolo aggiornato con "by Loop507"
    st.title("🎵 SoundWave Visualizer")
    st.caption("by Loop507 - Crea visualizzazioni video dei tuoi file audio")
    
    # Controlla disponibilità FFmpeg
    ffmpeg_available = check_ffmpeg()
    if not ffmpeg_available:
        st.warning("⚠️ FFmpeg non trovato. Solo anteprima disponibile.")
    
    # Controlli interfaccia
    st.sidebar.header("Impostazioni")
    
    mode = st.sidebar.selectbox("Modalità visualizzazione", list(VISUALIZATION_MODES.keys()))
    st.sidebar.caption(VISUALIZATION_MODES[mode])
    
    # Selettore FPS
    fps = st.sidebar.selectbox("Frame Rate (FPS)", FPS_OPTIONS, index=3)  # Default 30
    st.sidebar.caption(f"Fluidità: {fps} frame al secondo")
    
    color_preset = st.sidebar.selectbox("Schema colori", list(FREQUENCY_COLOR_PRESETS.keys()))
    colors = FREQUENCY_COLOR_PRESETS[color_preset]
    
    # Preview colori
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
    
    # Informazioni sui limiti
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Limiti:**")
    st.sidebar.caption(f"• File max: {MAX_FILE_SIZE // (1024*1024)} MB")
    st.sidebar.caption(f"• Durata max: {MAX_DURATION // 60:.0f} minuti")
    st.sidebar.caption("• Analisi: STFT Spectrum")
    
    # Contenuto principale
    uploaded = st.file_uploader("Carica file audio", type=["wav", "mp3", "m4a", "flac"])
    
    if uploaded:
        if not validate_audio_file(uploaded):
            return
        
        # Usa file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded.read())
            temp_audio = temp_file.name
        
        try:
            with st.spinner("Elaborazione audio..."):
                y, sr, duration = load_and_process_audio(temp_audio)
            
            if y is None:
                return
            
            with st.spinner("Analisi spettro STFT..."):
                features = generate_audio_features(y, sr, fps)
            
            if features is None:
                return
            
            # Informazioni audio
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Durata", f"{duration:.1f}s")
            with col2:
                st.metric("BPM", f"{features['tempo']:.1f}")
            with col3:
                st.metric("Sample Rate", f"{sr} Hz")
            with col4:
                st.metric("FPS Selezionati", f"{fps}")
            
            st.success("✅ Audio processato con spettro STFT!")
            
            # Anteprima
            st.subheader("🎭 Anteprima Visualizzazione")
            create_preview_visualization(features, mode, colors, resolution)
            
            # Generazione video
            if ffmpeg_available:
                st.subheader("🎬 Generazione Video")
                
                # Calcola frame totali senza limiti artificiali
                total_frames = int(duration * fps)
                estimated_time = total_frames / 100  # Stima approssimativa
                
                st.info(f"Video completo: {total_frames} frame a {fps} FPS (≈{estimated_time:.1f}s di elaborazione)")
                
                if st.button("🎬 Genera Video Completo", type="primary"):
                    with st.spinner(f"Generazione {total_frames} frame..."):
                        frames = []
                        progress_bar = st.progress(0)
                        
                        # Genera tutti i frame
                        for i in range(total_frames):
                            frame = generate_visualization_frame_pil(features, i, mode, colors, resolution, fps)
                            frames.append(frame)
                            
                            # Aggiorna progress ogni 100 frame
                            if i % 100 == 0 or i == total_frames - 1:
                                progress_bar.progress((i + 1) / total_frames)
                    
                    output_path = "output_visualization.mp4"
                    
                    with st.spinner("Creazione video finale..."):
                        create_video_from_images(frames, temp_audio, fps, output_path, resolution)
                    
                    if os.path.exists(output_path):
                        st.success("🎉 Video generato con successo!")
                        
                        # Mostra video
                        st.video(output_path)
                        
                        # Pulsante download
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "📥 Scarica Video",
                                f.read(),
                                file_name=f"soundwave_{mode.lower().replace(' ', '_')}_{fps}fps.mp4",
                                mime="video/mp4"
                            )
                        
                        # Pulizia
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
