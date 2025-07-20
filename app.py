# app.py - SoundWave Visualizer by Loop507 (Streamlit Cloud Compatible) - FIXED CONSISTENCY
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
import colorsys

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
    "Monochrome": {"high": "#FFFFFF", "mid": "#808080", "low": "#404040"},
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

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_visualization_data(features: Dict[str, Any], frame_idx: int, mode: str, fps: int) -> Dict[str, Any]:
    """
    NUOVA FUNZIONE CHIAVE: Genera i dati di visualizzazione in modo consistente
    sia per l'anteprima che per il video finale.
    """
    # Calcola indice temporale basato su FPS
    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['stft_magnitude'].shape[1] - 1)
    
    visualization_data = {
        'time_idx': time_idx,
        'mode': mode
    }
    
    if mode == "Classic Waveform":
        # Usa lo spettro STFT come base per il waveform
        stft_spec = features['stft_magnitude']
        
        # Media lungo le frequenze per creare effetto waveform
        waveform_full = np.mean(stft_spec, axis=0)
        
        # Finestra temporale per mostrare la progressione
        window_size = min(stft_spec.shape[1] // 4, 200)
        start_idx = max(0, time_idx - window_size)
        end_idx = min(stft_spec.shape[1], time_idx + 1)
        
        if end_idx > start_idx:
            waveform_section = waveform_full[start_idx:end_idx]
            time_section = np.linspace(start_idx, end_idx, len(waveform_section))
        else:
            waveform_section = np.array([waveform_full[time_idx]] if time_idx < len(waveform_full) else [0])
            time_section = np.array([time_idx])
        
        visualization_data.update({
            'waveform_data': waveform_section,
            'time_data': time_section,
            'full_waveform': waveform_full
        })
    
    elif mode == "Dense Matrix":
        stft_data = features['stft_magnitude']
        
        # Finestra temporale per la matrice
        window_size = min(100, stft_data.shape[1] // 10)
        start_idx = max(0, time_idx - window_size // 2)
        end_idx = min(stft_data.shape[1], start_idx + window_size)
        
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        
        # Dati della matrice per il frame corrente
        matrix_data = stft_data[:, start_idx:end_idx]
        
        visualization_data.update({
            'matrix_data': matrix_data,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'full_matrix': stft_data
        })
    
    elif mode == "Frequency Spectrum":
        if time_idx < features['freq_low'].shape[1]:
            # Finestra di visualizzazione
            window_size = min(200, features['freq_low'].shape[1])
            start_frame = max(0, time_idx - window_size)
            end_frame = min(features['freq_low'].shape[1], time_idx + 1)
            
            if end_frame > start_frame:
                # Calcola medie per ogni banda di frequenza
                low_data = np.mean(features['freq_low'][:, start_frame:end_frame], axis=0)
                mid_data = np.mean(features['freq_mid'][:, start_frame:end_frame], axis=0)
                high_data = np.mean(features['freq_high'][:, start_frame:end_frame], axis=0)
                time_frames = np.linspace(start_frame, end_frame, len(low_data))
            else:
                # Frame singolo
                low_data = np.array([np.mean(features['freq_low'][:, time_idx]) if time_idx < features['freq_low'].shape[1] else 0])
                mid_data = np.array([np.mean(features['freq_mid'][:, time_idx]) if time_idx < features['freq_mid'].shape[1] else 0])
                high_data = np.array([np.mean(features['freq_high'][:, time_idx]) if time_idx < features['freq_high'].shape[1] else 0])
                time_frames = np.array([time_idx])
            
            visualization_data.update({
                'low_freq_data': low_data,
                'mid_freq_data': mid_data,
                'high_freq_data': high_data,
                'time_frames': time_frames,
                'full_low': np.mean(features['freq_low'], axis=0),
                'full_mid': np.mean(features['freq_mid'], axis=0),
                'full_high': np.mean(features['freq_high'], axis=0)
            })
    
    return visualization_data

def generate_visualization_frame_pil(features: Dict[str, Any], frame_idx: int, mode: str, 
                                   colors: Dict[str, str], resolution: Tuple[int, int], fps: int) -> Image.Image:
    """Generate a visualization frame using PIL - USA I DATI CONSISTENTI."""
    width, height = resolution
    img = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img)
    
    # CHIAVE: Usa la stessa funzione per generare i dati
    vis_data = generate_visualization_data(features, frame_idx, mode, fps)
    
    if mode == "Classic Waveform":
        waveform_section = vis_data['waveform_data']
        
        if len(waveform_section) > 0:
            # Coordinate X distribuite sulla larghezza
            x_coords = np.linspace(0, width, len(waveform_section))
            
            # Crea punti per la forma riempita
            points = []
            for x, intensity in zip(x_coords, waveform_section):
                y = height - int(intensity * height * 0.8)
                points.append((int(x), y))
            
            # Chiudi la forma
            if points:
                points.append((width, height))
                points.append((0, height))
                
                # Riempi e contorna
                draw.polygon(points, fill=colors['mid'])
                draw.polygon(points, outline=colors['high'])
    
    elif mode == "Dense Matrix":
        matrix_data = vis_data['matrix_data']
        
        if matrix_data.size > 0:
            freq_bins = min(matrix_data.shape[0], height // 2)
            time_bins = matrix_data.shape[1]
            
            cell_width = max(1, width // time_bins)
            cell_height = max(1, height // freq_bins)
            
            for freq_idx in range(freq_bins):
                for time_offset in range(time_bins):
                    # Inverti frequenze (basse in basso)
                    intensity = matrix_data[freq_bins - 1 - freq_idx, time_offset]
                    
                    if intensity > 0.05:  # Soglia per evitare il nero
                        intensity_scaled = min(1.0, max(0.0, intensity))
                        
                        # Colormap hot identica all'anteprima
                        if intensity_scaled < 0.33:
                            factor = intensity_scaled / 0.33
                            red = int(255 * factor)
                            green = 0
                            blue = 0
                        elif intensity_scaled < 0.66:
                            factor = (intensity_scaled - 0.33) / 0.33
                            red = 255
                            green = int(255 * factor)
                            blue = 0
                        else:
                            factor = (intensity_scaled - 0.66) / 0.34
                            red = 255
                            green = 255
                            blue = int(255 * factor)
                        
                        color = f"#{red:02x}{green:02x}{blue:02x}"
                        
                        x1 = time_offset * cell_width
                        y1 = freq_idx * cell_height
                        x2 = min(width, x1 + cell_width)
                        y2 = min(height, y1 + cell_height)
                        
                        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    elif mode == "Frequency Spectrum":
        low_data = vis_data.get('low_freq_data', np.array([]))
        mid_data = vis_data.get('mid_freq_data', np.array([]))
        high_data = vis_data.get('high_freq_data', np.array([]))
        
        if len(low_data) > 0:
            x_coords = np.linspace(0, width, len(low_data))
            
            def draw_frequency_line(data, color, line_width=4):
                if len(data) > 1:
                    points = []
                    for x, intensity in zip(x_coords, data):
                        y = height - int(intensity * height * 0.8)
                        points.append((int(x), y))
                    
                    # Linea spessa mediante offset multipli
                    for i in range(len(points) - 1):
                        x1, y1 = points[i]
                        x2, y2 = points[i + 1]
                        
                        for offset_y in range(-line_width//2, line_width//2 + 1):
                            for offset_x in range(-1, 2):
                                draw.line([x1 + offset_x, y1 + offset_y, 
                                         x2 + offset_x, y2 + offset_y], 
                                        fill=color, width=1)
            
            # Disegna le tre linee con i colori configurati
            draw_frequency_line(low_data, colors['low'], 5)
            draw_frequency_line(mid_data, colors['mid'], 4)
            draw_frequency_line(high_data, colors['high'], 3)
    
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
                               resolution: Tuple[int, int], fps: int = 30) -> None:
    """
    FUNZIONE COMPLETAMENTE RISCRITTA: Ora genera l'anteprima usando PIL
    per garantire perfetta consistenza con il video finale.
    """
    st.subheader("🎭 Anteprima Statica")
    
    # Genera alcuni frame di esempio per mostrare l'evoluzione
    sample_frames = [
        int(0.1 * features['duration'] * fps),  # 10%
        int(0.3 * features['duration'] * fps),  # 30%
        int(0.5 * features['duration'] * fps),  # 50%
        int(0.7 * features['duration'] * fps),  # 70%
        int(0.9 * features['duration'] * fps)   # 90%
    ]
    
    # Risoluzione ridotta per l'anteprima (più veloce)
    preview_resolution = (640, 360) if resolution[0] > resolution[1] else (360, 640)
    
    col1, col2, col3 = st.columns(3)
    
    # Frame centrale (50%)
    with col2:
        st.caption("🎯 Frame centrale (50%)")
        center_frame = generate_visualization_frame_pil(
            features, sample_frames[2], mode, colors, preview_resolution, fps
        )
        st.image(center_frame, use_column_width=True)
    
    # Frame iniziale e finale
    with col1:
        st.caption("🚀 Inizio (10%)")
        start_frame = generate_visualization_frame_pil(
            features, sample_frames[0], mode, colors, preview_resolution, fps
        )
        st.image(start_frame, use_column_width=True)
    
    with col3:
        st.caption("🏁 Fine (90%)")
        end_frame = generate_visualization_frame_pil(
            features, sample_frames[4], mode, colors, preview_resolution, fps
        )
        st.image(end_frame, use_column_width=True)
    
    # Anteprima evoluzione temporale
    st.subheader("📊 Evoluzione Temporale")
    
    # Genera grafico matplotlib SOLO per mostrare i dati audio
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if mode == "Classic Waveform":
        # Mostra il waveform completo come riferimento
        vis_data = generate_visualization_data(features, sample_frames[2], mode, fps)
        full_waveform = vis_data['full_waveform']
        time_axis = np.linspace(0, features['duration'], len(full_waveform))
        
        ax.fill_between(time_axis, 0, full_waveform, color=colors['mid'], alpha=0.7, label='STFT Waveform')
        ax.plot(time_axis, full_waveform, color=colors['high'], linewidth=1)
        ax.set_title("Classic Waveform - Dati Audio Originali")
        
    elif mode == "Dense Matrix":
        # Mostra spettrogramma completo
        vis_data = generate_visualization_data(features, sample_frames[2], mode, fps)
        full_matrix = vis_data['full_matrix']
        
        im = ax.imshow(full_matrix, aspect='auto', origin='lower', 
                      cmap='hot', extent=[0, features['duration'], 0, full_matrix.shape[0]])
        ax.set_title("Dense Matrix - Spettrogramma STFT Completo")
        plt.colorbar(im, ax=ax, label='Intensità')
        
    elif mode == "Frequency Spectrum":
        # Mostra tutte e tre le bande
        vis_data = generate_visualization_data(features, sample_frames[2], mode, fps)
        time_axis = np.linspace(0, features['duration'], len(vis_data['full_low']))
        
        ax.plot(time_axis, vis_data['full_low'], color=colors['low'], label='Basse', linewidth=3)
        ax.plot(time_axis, vis_data['full_mid'], color=colors['mid'], label='Medie', linewidth=3)
        ax.plot(time_axis, vis_data['full_high'], color=colors['high'], label='Acute', linewidth=3)
        ax.legend()
        ax.set_title("Frequency Spectrum - Bande Complete")
    
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Intensità")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.success("✅ L'anteprima mostra esattamente come apparirà il video finale!")

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
    st.sidebar.header("🎨 Impostazioni Visualizzazione")
    
    mode = st.sidebar.selectbox("Modalità visualizzazione", list(VISUALIZATION_MODES.keys()))
    st.sidebar.caption(VISUALIZATION_MODES[mode])
    
    # Selettore FPS
    fps = st.sidebar.selectbox("Frame Rate (FPS)", FPS_OPTIONS, index=3)  # Default 30
    st.sidebar.caption(f"Fluidità: {fps} frame al secondo")
    
    # MIGLIORATO: Sistema colori più flessibile
    st.sidebar.subheader("🌈 Schema Colori")
    color_preset = st.sidebar.selectbox("Preset colori", list(FREQUENCY_COLOR_PRESETS.keys()))
    
    # Se è Custom, mostra color picker
    if color_preset == "Custom":
        colors = {
            "high": st.sidebar.color_picker("Colore frequenze acute", "#FFFF00"),
            "mid": st.sidebar.color_picker("Colore frequenze medie", "#00FF00"), 
            "low": st.sidebar.color_picker("Colore frequenze basse", "#FF0000")
        }
    else:
        colors = FREQUENCY_COLOR_PRESETS[color_preset].copy()
        
        # Possibilità di modificare i preset
        with st.sidebar.expander("✏️ Modifica preset"):
            colors["high"] = st.color_picker("Frequenze acute", colors["high"])
            colors["mid"] = st.color_picker("Frequenze medie", colors["mid"])
            colors["low"] = st.color_picker("Frequenze basse", colors["low"])
    
    # Preview colori migliorato
    st.sidebar.markdown("**Preview colori:**")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.markdown(f'<div style="background-color:{colors["high"]}; height:25px; border-radius:5px; border:1px solid #333;"></div>', unsafe_allow_html=True)
        st.caption("🎵 Acute")
    with col2:
        st.markdown(f'<div style="background-color:{colors["mid"]}; height:25px; border-radius:5px; border:1px solid #333;"></div>', unsafe_allow_html=True)
        st.caption("🎤 Medie")
    with col3:
        st.markdown(f'<div style="background-color:{colors["low"]}; height:25px; border-radius:5px; border:1px solid #333;"></div>', unsafe_allow_html=True)
        st.caption("🥁 Basse")
    
    resolution_format = st.sidebar.selectbox("Formato video", list(FORMAT_RESOLUTIONS.keys()))
    resolution = FORMAT_RESOLUTIONS[resolution_format]
    st.sidebar.caption(f"Risoluzione: {resolution[0]}x{resolution[1]}")
    
    # Informazioni sui limiti
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Specifiche Tecniche:**")
    st.sidebar.caption(f"• File max: {MAX_FILE_SIZE // (1024*1024)} MB")
    st.sidebar.caption(f"• Durata max: {MAX_DURATION // 60:.0f} minuti")
    st.sidebar.caption("• Analisi: STFT Spectrum")
    st.sidebar.caption("• Qualità: Alta definizione")
    
    # Contenuto principale
    st.header("📁 Carica Audio")
    uploaded = st.file_uploader("Scegli file audio", type=["wav", "mp3", "m4a", "flac", "aac", "ogg"])
    
    if uploaded:
        if not validate_audio_file(uploaded):
            return
        
        # Usa file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded.read())
            temp_audio = temp_file.name
        
        try:
            with st.spinner("🎵 Elaborazione audio..."):
                y, sr, duration = load_and_process_audio(temp_audio)
            
            if y is None:
                return
            
            with st.spinner("📊 Analisi spettro STFT..."):
                features = generate_audio_features(y, sr, fps)
            
            if features is None:
                return
            
            # Informazioni audio con layout migliorato
            st.header("📊 Informazioni Audio")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("⏱️ Durata", f"{duration:.1f}s")
            with col2:
                st.metric("🥁 BPM", f"{features['tempo']:.1f}")
            with col3:
                st.metric("📡 Sample Rate", f"{sr} Hz")
            with col4:
                st.metric("🎬 FPS Output", f"{fps}")
            with col5:
                st.metric("📐 Risoluzione", f"{resolution[0]}×{resolution[1]}")
            
            st.success("✅ Audio processato con successo! Spettro STFT generato.")
            
            # Anteprima CORRETTA
            st.header("🎭 Anteprima Visualizzazione")
            st.info("💡 Questa anteprima mostra ESATTAMENTE come apparirà il video finale")
            create_preview_visualization(features, mode, colors, resolution, fps)

# Generazione video
            if ffmpeg_available:
                st.header("🎬 Generazione Video")
                
                if st.button("🚀 Genera Video", type="primary", use_container_width=True):
                    total_frames = int(duration * fps)
                    
                    if total_frames > 10000:  # Limite sicurezza (~5 min a 30fps)
                        st.warning(f"⚠️ Video lungo: {total_frames} frame. Potrebbero servire diversi minuti.")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        with st.spinner("🎨 Generazione frame..."):
                            frames = []
                            
                            for frame_idx in range(total_frames):
                                # Aggiorna progress ogni 50 frame
                                if frame_idx % 50 == 0:
                                    progress = frame_idx / total_frames
                                    progress_bar.progress(progress)
                                    status_text.text(f"Generando frame {frame_idx + 1}/{total_frames} ({progress*100:.1f}%)")
                                
                                # Genera frame usando PIL (identico all'anteprima)
                                frame = generate_visualization_frame_pil(
                                    features, frame_idx, mode, colors, resolution, fps
                                )
                                frames.append(frame)
                                
                                # Pulizia memoria ogni 100 frame
                                if frame_idx % 100 == 0:
                                    gc.collect()
                        
                        # Creazione video
                        with st.spinner("🎞️ Assemblaggio video finale..."):
                            output_path = tempfile.mktemp(suffix=".mp4")
                            create_video_from_images(frames, temp_audio, fps, output_path, resolution)
                            
                            if os.path.exists(output_path):
                                progress_bar.progress(1.0)
                                status_text.text("✅ Video completato!")
                                
                                # Download
                                with open(output_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Scarica Video",
                                        data=f.read(),
                                        file_name=f"soundwave_{mode.lower().replace(' ', '_')}_{fps}fps.mp4",
                                        mime="video/mp4",
                                        type="primary",
                                        use_container_width=True
                                    )
                                
                                st.success(f"🎉 Video generato: {total_frames} frame a {fps} FPS!")
                                
                                # Statistiche finali
                                st.info(f"📈 **Statistiche**: Modalità {mode} • {resolution[0]}×{resolution[1]} • Schema {color_preset}")
                                
                                cleanup_files(output_path)
                            else:
                                st.error("❌ Errore nella generazione del video")
                    
                    except Exception as e:
                        st.error(f"❌ Errore durante la generazione: {e}")
                    finally:
                        progress_bar.empty()
                        status_text.empty()
                        gc.collect()
            else:
                st.header("🎬 Generazione Video")
                st.error("❌ FFmpeg richiesto per la generazione video")
                st.info("💡 Installa FFmpeg per abilitare l'esportazione video")
        
        except Exception as e:
            st.error(f"❌ Errore generale: {e}")
        finally:
            cleanup_files(temp_audio)
    
    else:
        # Schermata iniziale migliorata
        st.header("🎯 Come iniziare")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📤 **1. Carica Audio**
            - Formati: MP3, WAV, FLAC, M4A
            - Max 200 MB / 30 minuti  
            - Qualità ottimale: 44.1kHz
            """)
        
        with col2:
            st.markdown("""
            ### 🎨 **2. Personalizza**
            - 3 modalità visualizzazione
            - Colori e preset personalizzabili
            - Risoluzioni: 16:9, 9:16, 1:1, 4:3
            """)
        
        with col3:
            st.markdown("""
            ### 🚀 **3. Esporta**
            - Frame rate fino a 30 FPS
            - Video HD di alta qualità
            - Anteprima in tempo reale
            """)
        
        st.markdown("---")
        
        # Esempi visualizzazioni
        st.subheader("🎭 Modalità Disponibili")
        
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            st.markdown("**🌊 Classic Waveform**")
            st.caption("Forma d'onda dinamica verticale con riempimento colorato")
        
        with example_col2:
            st.markdown("**🔳 Dense Matrix**")
            st.caption("Spettrogramma a matrice con colormap hot per frequenze")
        
        with example_col3:
            st.markdown("**📊 Frequency Spectrum**")
            st.caption("Tre linee per basse, medie e acute frequenze")
        
        # Footer
        st.markdown("---")
        st.markdown("**🛠️ Tecnologie**: STFT Analysis • FFmpeg • PIL Rendering • Streamlit")
        st.caption("Sviluppato da Loop507 - Visualizzatore audio avanzato per content creator")

if __name__ == "__main__":
    main()
