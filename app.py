# app.py - SoundWave Visualizer by Loop507 - ARTISTIC EDITION (Optimized)
import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageFont # AGGIUNTO ImageFont
import io
from typing import Tuple, Optional, Dict, Any
import colorsys
import math
import random
import sys

# ==========================================
# GESTIONE DELLE DIPENDENZE - ORA GESTITA TRAMITE requirements.txt
# La sezione `install_and_import` è stata rimossa.
# Assicurati di avere un file requirements.txt nella stessa directory con:
# streamlit
# numpy>=1.26.0
# librosa>=0.10.0
# matplotlib>=3.7.0
# Pillow>=10.0.0
# scipy>=1.10.0
# ==========================================

# Costanti - AGGIORNATE
MAX_DURATION: float = 1800  # 30 minuti
MIN_DURATION: float = 1.0
MAX_FILE_SIZE: int = 200 * 1024 * 1024  # 200 MB

# Opzioni FPS disponibili
FPS_OPTIONS: list = [5, 10, 20, 30]

FORMAT_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:3": (800, 600)
}

# NUOVI STILI ARTISTICI - AGGIUNTA "Barcode Visualizer"
ARTISTIC_STYLES: Dict[str, str] = {
    "Particle System": "🌟 Particelle che danzano con la musica",
    "Circular Spectrum": "⭕ Spettro radiale rotante",
    "3D Waveforms": "🌊 Onde pseudo-3D dinamiche",
    "Fluid Dynamics": "💧 Simulazione fluidi realistici",
    "Geometric Patterns": "🔷 Forme geometriche animate",
    "Neural Network": "🧠 Rete neurale pulsante",
    "Galaxy Spiral": "🌌 Spirale galattica in movimento",
    "Lightning Storm": "⚡ Tempesta elettrica musicale",
    "Barcode Visualizer": "📶 Barre che reagiscono alla musica" # Nuovo stile
}

# INTENSITÀ MOVIMENTO - AGGIORNATA
MOVEMENT_INTENSITY: Dict[str, float] = {
    "Soft": 0.2,
    "Medio": 1.0,
    "Hard": 3.0
}

# TEMI COLORE ARTISTICI - MANTENUTI PER GLI ALTRI STILI, MA USEREMO SCELTA PERSONALIZZATA PER PARTICELLE
ARTISTIC_COLOR_THEMES: Dict[str, Dict[str, Any]] = {
    "Neon Cyber": {
        "colors": ["#FF0080", "#00FF80", "#8000FF", "#FF8000"],
        "background": "#000015",
        "style": "neon"
    },
    "Ocean Deep": {
        "colors": ["#001a2e", "#0074d9", "#39cccc", "#85c1e9"],
        "background": "#000a1a",
        "style": "fluid"
    },
    "Sunset Fire": {
        "colors": ["#ff6b35", "#f7931e", "#ffd23f", "#ff0040"],
        "background": "#1a0000",
        "style": "fire"
    },
    "Aurora": {
        "colors": ["#00ff88", "#0088ff", "#8800ff", "#ff0088"],
        "background": "#001122",
        "style": "glow"
    },
    "Minimal White": {
        "colors": ["#ffffff", "#e0e0e0", "#c0c0c0", "#a0a0a0"],
        "background": "#000000",
        "style": "clean"
    },
    "Galaxy": {
        "colors": ["#4a0e4e", "#7209b7", "#a663cc", "#4cc9f0"],
        "background": "#0d0221",
        "style": "space"
    }
}

# Costanti per la definizione delle bande di frequenza (Hz)
BASS_HZ_RANGE = (20, 500)
MID_HZ_RANGE = (500, 4000)
HIGH_HZ_RANGE = (4000, 11000) # Fino al limite di Nyquist per SR=22050

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
def generate_enhanced_audio_features(y: np.ndarray, sr: int, fps: int) -> Optional[Dict[str, Any]]:
    """Generate enhanced audio features for artistic visualization."""
    try:
        duration = len(y) / sr

        # STFT Analysis multipla per diversi livelli di dettaglio
        hop_length = 512
        n_fft = 2048

        # Spettro principale
        stft = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        stft_norm = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-9)

        # Analisi dettagliata delle frequenze (bande definite in Hz)
        freq_bins_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Trova gli indici dei bin per le bande di frequenza definite in Hz
        bass_start_bin = np.searchsorted(freq_bins_hz, BASS_HZ_RANGE[0])
        bass_end_bin = np.searchsorted(freq_bins_hz, BASS_HZ_RANGE[1], side='right') - 1
        bass_start_bin = max(0, bass_start_bin)
        bass_end_bin = min(len(freq_bins_hz) - 1, bass_end_bin)
        freq_bass = stft_norm[bass_start_bin : bass_end_bin + 1, :]

        mid_start_bin = np.searchsorted(freq_bins_hz, MID_HZ_RANGE[0])
        mid_end_bin = np.searchsorted(freq_bins_hz, MID_HZ_RANGE[1], side='right') - 1
        mid_start_bin = max(0, mid_start_bin)
        mid_end_bin = min(len(freq_bins_hz) - 1, mid_end_bin)
        freq_high_mid = stft_norm[mid_start_bin : mid_end_bin + 1, :]

        high_start_bin = np.searchsorted(freq_bins_hz, HIGH_HZ_RANGE[0])
        high_end_bin = np.searchsorted(freq_bins_hz, HIGH_HZ_RANGE[1], side='right') - 1
        high_start_bin = max(0, high_start_bin)
        high_end_bin = min(len(freq_bins_hz) - 1, high_end_bin)
        freq_brilliance = stft_norm[high_start_bin : high_end_bin + 1, :]


        # Chromagram per tonalità
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma_norm = (chroma - chroma.min()) / (chroma.max() - chroma.min() + 1e-9)

        # Spectral features avanzati
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

        # Normalizzazione features
        centroid_norm = (spectral_centroid - spectral_centroid.min()) / (spectral_centroid.max() - spectral_centroid.min() + 1e-9)
        rolloff_norm = (spectral_rolloff - spectral_rolloff.min()) / (spectral_rolloff.max() - spectral_rolloff.min() + 1e-9)
        bandwidth_norm = (spectral_bandwidth - spectral_bandwidth.min()) / (spectral_bandwidth.max() - spectral_bandwidth.min() + 1e-9)
        zcr_norm = (zero_crossing_rate - zero_crossing_rate.min()) / (zero_crossing_rate.max() - zero_crossing_rate.min() + 1e-9)

        # RMS Energy e onset detection
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_norm = np.clip(rms / np.max(rms) if np.max(rms) > 0 else rms, 0, 1)

        # Onset detection per eventi musicali
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_norm = (onset_strength - onset_strength.min()) / (onset_strength.max() - onset_strength.min() + 1e-9)

        # Tempo e beat tracking
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)
        except Exception:
            tempo = 120.0
            beats = np.array([])

        # MFCC per timbro
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_norm = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-9)

        return {
            # Spettri base
            'stft_magnitude': stft_norm,
            'chroma': chroma_norm,
            'mfcc': mfcc_norm,

            # Bande frequenza dettagliate (solo quelle definite)
            'freq_bass': freq_bass,
            'freq_high_mid': freq_high_mid,
            'freq_brilliance': freq_brilliance,
            
            # Range Hz per le bande principali (ora più precise)
            'bass_hz_range': (freq_bins_hz[bass_start_bin], freq_bins_hz[bass_end_bin]),
            'mid_hz_range': (freq_bins_hz[mid_start_bin], freq_bins_hz[mid_end_bin]),
            'high_hz_range': (freq_bins_hz[high_start_bin], freq_bins_hz[high_end_bin]),

            # Features spettrali
            'spectral_centroid': centroid_norm,
            'spectral_rolloff': rolloff_norm,
            'spectral_bandwidth': bandwidth_norm,
            'zero_crossing_rate': zcr_norm,

            # Energy e dinamica
            'rms_energy': rms_norm,
            'onset_strength': onset_norm,
            'onset_frames': onset_frames,

            # Ritmo
            'beats': beats,
            'tempo': tempo,

            # Parametri tecnici
            'hop_length': hop_length,
            'sr': sr,
            'duration': duration,
            'magnitude_raw': magnitude,
            'n_fft': n_fft
        }
    except Exception as e:
        st.error(f"Errore feature avanzate: {e}")
        return None

def get_time_idx(features: Dict[str, Any], frame_idx: int, fps: int) -> int:
    """Calcola l'indice temporale STFT/feature per un dato frame video."""
    current_time = frame_idx / fps
    time_idx = librosa.time_to_frames(current_time, sr=features['sr'], hop_length=features['hop_length'])
    return min(time_idx, features['rms_energy'].shape[0] - 1)

def draw_text_on_frame(draw: ImageDraw.Draw, resolution: Tuple[int, int], custom_text: str,
                       text_font: Any, text_color: str, text_position: str):
    """Disegna testo su un frame."""
    width, height = resolution
    text_rgb = hex_to_rgb(text_color)
    
    if not custom_text:
        return # Non disegnare se il testo è vuoto

    # Calcola la dimensione del testo
    # text_bbox è (left, top, right, bottom)
    text_bbox = draw.textbbox((0, 0), custom_text, font=text_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calcola la posizione
    x_text, y_text = 0, 0
    padding = 10 # Spaziatura dai bordi

    if text_position == "Top-Left":
        x_text, y_text = padding, padding
    elif text_position == "Top-Center":
        x_text = (width - text_width) // 2
        y_text = padding
    elif text_position == "Top-Right":
        x_text = width - text_width - padding
        y_text = padding
    elif text_position == "Bottom-Left":
        x_text = padding
        y_text = height - text_height - padding
    elif text_position == "Bottom-Center":
        x_text = (width - text_width) // 2
        y_text = height - text_height - padding
    elif text_position == "Bottom-Right":
        x_text = width - text_width - padding
        y_text = height - text_height - padding
    elif text_position == "Center":
        x_text = (width - text_width) // 2
        y_text = (height - text_height) // 2

    draw.text((x_text, y_text), custom_text, font=text_font, fill=text_rgb)


# --- FUNZIONI DI STILE MODIFICATE PER SFONDO E TESTO ---

def create_particle_system(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                         background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                         text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)
        onset_strength = features['onset_strength'][time_idx]

        bass_energy = np.mean(features['freq_bass'][:, time_idx])
        mid_energy = np.mean(features['freq_high_mid'][:, time_idx])
        high_energy = np.mean(features['freq_brilliance'][:, time_idx])

        num_particles = int(200 + effective_energy * 500 * intensity)

        for i in range(num_particles):
            angle = (i / num_particles) * 2 * np.pi + frame_idx * 0.08 * intensity
            base_radius = min(width, height) * 0.15
            radius_variation = (bass_energy * 0.5 + mid_energy * 0.3 + high_energy * 0.2) * min(width, height) * 0.4
            radius = base_radius + radius_variation * (1 + np.sin(angle * 5 + frame_idx * 0.15))

            center_x, center_y = width // 2, height // 2
            x = int(center_x + radius * np.cos(angle) * (1 + effective_energy * 0.2))
            y = int(center_y + radius * np.sin(angle) * (1 + effective_energy * 0.2))

            particle_size = int(4 + onset_strength * 25 + effective_energy * 15 * intensity)
            particle_size = max(1, particle_size)

            colors = theme['colors']
            if len(colors) == 1:
                color = colors[0]
            else:
                if bass_energy > mid_energy and bass_energy > high_energy:
                    color = colors[0]
                elif mid_energy > high_energy:
                    color = colors[1 % len(colors)]
                else:
                    color = colors[2 % len(colors)]
            
            alpha = int(150 + effective_energy * 105)

            for glow_radius in range(particle_size + 6, particle_size - 1, -1):
                glow_alpha = max(10, alpha // (glow_radius - particle_size + 2))
                if glow_radius > particle_size:
                    r, g, b = hex_to_rgb(color)
                    h, s, v = colorsys.rgb_to_hsv(r / 255., g / 255., b / 255.)
                    v_glow = min(1.0, v * 1.5)
                    s_glow = min(1.0, s * 0.8)
                    r_glow, g_glow, b_glow = colorsys.hsv_to_rgb(h, s_glow, v_glow)
                    r_glow_int = int(r_glow * 255)
                    g_glow_int = int(g_glow * 255)
                    b_glow_int = int(b_glow * 255)
                    final_glow_color = (r_glow_int, g_glow_int, b_glow_int, glow_alpha)
                else:
                    final_glow_color = (*hex_to_rgb(color), alpha)

                draw.ellipse([x - glow_radius, y - glow_radius,
                            x + glow_radius, y + glow_radius], fill=final_glow_color)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')


def create_circular_spectrum(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                           theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                           background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                           text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) * 0.4

        spectrum_slice = features['stft_magnitude'][:, time_idx]
        n_bins = len(spectrum_slice)

        rotation_offset = frame_idx * 0.02 * intensity

        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        for i in range(0, n_bins, max(1, n_bins // 120)):
            if i < len(spectrum_slice):
                magnitude = spectrum_slice[i]

                angle = (i / n_bins) * 2 * np.pi + rotation_offset

                inner_radius = max_radius * 0.3
                outer_radius = inner_radius + magnitude * max_radius * 0.6 * intensity * effective_energy

                x1 = int(center_x + inner_radius * np.cos(angle))
                y1 = int(center_y + inner_radius * np.sin(angle))
                x2 = int(center_x + outer_radius * np.cos(angle))
                y2 = int(center_y + outer_radius * np.sin(angle))

                colors = theme['colors']
                if len(colors) == 1:
                    color = colors[0]
                else:
                    if i < n_bins // 3:
                        color = colors[0 % len(colors)]
                    elif i < 2 * n_bins // 3:
                        color = colors[1 % len(colors)]
                    else:
                        color = colors[2 % len(colors)]
                
                line_width = max(1, int(magnitude * 5 * effective_energy + 1))

                draw.line([x1, y1, x2, y2], fill=color, width=line_width)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')

def create_3d_waveforms(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                       theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                       background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                       text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        window_size = min(100, features['rms_energy'].shape[0] // 4)
        start_idx = max(0, time_idx - window_size // 2)
        end_idx = min(features['rms_energy'].shape[0], start_idx + window_size)

        if end_idx > start_idx:
            waveform_data = features['rms_energy'][start_idx:end_idx]

            layers = 5
            for layer in range(layers):
                layer_offset_y = layer * 20
                
                if layer == 0:
                    base_color = theme['colors'][0]
                elif layer == 1:
                    base_color = theme['colors'][1 % len(theme['colors'])]
                else:
                    base_color = theme['colors'][2 % len(theme['colors'])]

                points = []
                for i, amplitude in enumerate(waveform_data):
                    x = int((i / len(waveform_data)) * width)
                    base_y = height // 2 + layer_offset_y
                    wave_y = int(amplitude * height * 0.3 * intensity * np.sin(frame_idx * 0.1 + layer) * effective_energy)
                    y = base_y - wave_y

                    points.append((x, y))

                if len(points) > 1:
                    fill_points = points + [(width, height), (0, height)]

                    r, g, b = hex_to_rgb(base_color)
                    dark_factor = 1.0 - (layer / (layers * 1.5))
                    r = int(r * dark_factor)
                    g = int(g * dark_factor)
                    b = int(b * dark_factor)
                    layer_color = (r, g, b, 255) # Use RGBA directly for fill in PIL

                    draw.polygon(fill_points, fill=layer_color)

                    line_thickness = max(1, int(2 * effective_energy))
                    for i in range(len(points) - 1):
                        draw.line([points[i], points[i + 1]], fill=base_color, width=line_thickness)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')

def create_fluid_dynamics(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                         background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                         text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        num_waves = 8

        for wave_idx in range(num_waves):
            wave_phase = frame_idx * 0.05 * intensity + wave_idx * np.pi / 4
            amplitude = effective_energy * height * 0.2 * intensity
            frequency = 0.01 + wave_idx * 0.005

            points = []
            for x in range(0, width, 5):
                y_base = height // 2 + wave_idx * 30 - num_waves * 15
                y_offset = amplitude * np.sin(x * frequency + wave_phase)
                y = int(y_base + y_offset)
                y = max(0, min(height - 1, y))
                points.append((x, y))

            if wave_idx % 3 == 0:
                color = theme['colors'][0]
            elif wave_idx % 3 == 1:
                color = theme['colors'][1 % len(theme['colors'])]
            else:
                color = theme['colors'][2 % len(theme['colors'])]

            if len(points) > 2:
                bottom_points = [(width, height), (0, height)]
                wave_polygon = points + bottom_points

                r, g, b = hex_to_rgb(color)
                alpha = int(100 + effective_energy * 155) # Use 255 for full opacity
                wave_color_with_alpha = (r, g, b, alpha)

                draw.polygon(wave_polygon, fill=wave_color_with_alpha)

                line_thickness = max(1, int(2 * effective_energy))
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i + 1]], fill=color, width=line_thickness)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')

def create_geometric_patterns(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                            theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                            background_image: Optional[Image.Image], custom_text: str, text_font: Any, # <--- Corretto qui!
                            text_color: str, text_position: str) -> Image.Image: # <--- E qui!
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)
        center_x, center_y = width // 2, height // 2

        num_rings = 6
        base_radius = min(width, height) * 0.05

        for ring in range(num_rings):
            radius = base_radius + ring * (min(width, height) * 0.08)
            radius *= (1 + effective_energy * intensity * 0.5)

            rotation = frame_idx * 0.02 * intensity + ring * 0.5

            sides = 3 + ring

            points = []
            for side in range(sides):
                angle = (side / sides) * 2 * np.pi + rotation
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append((x, y))

            if ring % 3 == 0:
                color = theme['colors'][0]
            elif ring % 3 == 1:
                color = theme['colors'][1 % len(theme['colors'])]
            else:
                color = theme['colors'][2 % len(theme['colors'])]

            if len(points) > 2:
                outline_width = max(1, int(3 * effective_energy))

                if ring % 2 == 0:
                    r, g, b = hex_to_rgb(color)
                    alpha = int(100 + effective_energy * 155)
                    fill_color_rgb = (r, g, b, alpha)
                    draw.polygon(points, fill=fill_color_rgb)
                else:
                    draw.polygon(points, outline=color, width=outline_width)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')

def create_neural_network(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                         background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                         text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        nodes = []
        grid_size = 8

        for i in range(grid_size):
            for j in range(grid_size):
                x = int((i + 1) / (grid_size + 1) * width)
                y = int((j + 1) / (grid_size + 1) * height)

                activation = effective_energy + 0.3 * np.sin(frame_idx * 0.1 + i + j)
                activation = max(0, min(1, activation))

                nodes.append((x, y, activation))

        for i, (x1, y1, act1) in enumerate(nodes):
            for j, (x2, y2, act2) in enumerate(nodes[i+1:], i+1):
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                if distance < min(width, height) * 0.2:
                    connection_strength = (act1 + act2) / 2 * intensity

                    if connection_strength > 0.3:
                        line_width = max(1, int(connection_strength * 3))

                        if connection_strength < 0.5:
                            color = theme['colors'][0]
                        elif connection_strength < 0.75:
                            color = theme['colors'][1 % len(theme['colors'])]
                        else:
                            color = theme['colors'][2 % len(theme['colors'])]

                        draw.line([x1, y1, x2, y2], fill=color, width=line_width)

        for x, y, activation in nodes:
            if activation > 0.2:
                node_size = int(3 + activation * 8 * intensity)
                if activation < 0.5:
                    color = theme['colors'][0]
                elif activation < 0.75:
                    color = theme['colors'][1 % len(theme['colors'])]
                else:
                    color = theme['colors'][2 % len(theme['colors'])]

                draw.ellipse([x-node_size, y-node_size, x+node_size, y+node_size], fill=color)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')

def create_galaxy_spiral(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                        theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                        background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                        text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)
        center_x, center_y = width // 2, height // 2

        num_arms = 3
        points_per_arm = 100

        for arm in range(num_arms):
            arm_offset = (arm / num_arms) * 2 * np.pi

            points = []
            for i in range(points_per_arm):
                t = i / points_per_arm * 6 * np.pi
                radius = (i / points_per_arm) * min(width, height) * 0.4
                radius *= (1 + effective_energy * intensity * 0.3)
                angle = t + arm_offset + frame_idx * 0.02 * intensity

                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))

                points.append((x, y))

            if arm == 0:
                color = theme['colors'][0]
            elif arm == 1:
                color = theme['colors'][1 % len(theme['colors'])]
            else:
                color = theme['colors'][2 % len(theme['colors'])]

            if len(points) > 1:
                for i in range(len(points) - 1):
                    intensity_factor = 1 - (i / len(points))
                    line_width = max(1, int(2 + effective_energy * 3 * intensity_factor))
                    draw.line([points[i], points[i + 1]], fill=color, width=line_width)

            star_density = int(10 + effective_energy * 20)
            for star in range(star_density):
                point_idx = random.randint(0, len(points) - 1)
                if point_idx < len(points):
                    star_x, star_y = points[point_idx]
                    star_x += random.randint(-5, 5)
                    star_y += random.randint(-5, 5)

                    star_size = random.randint(1, 3)
                    draw.ellipse([star_x-star_size, star_y-star_size,
                                star_x+star_size, star_y+star_size], fill=color)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')

def create_lightning_storm(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                          theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                          background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                          text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)
        onset_strength = features['onset_strength'][time_idx]

        num_lightning = int(1 + onset_strength * 8 + effective_energy * 5)

        for lightning in range(num_lightning):
            start_x = random.randint(0, width)
            start_y = random.randint(0, height // 4)

            end_x = start_x + random.randint(-width//4, width//4)
            end_y = random.randint(3*height//4, height)

            current_x, current_y = start_x, start_y

            segments = 8 + int(effective_energy * 12)

            for segment in range(segments):
                progress = (segment + 1) / segments

                target_x = start_x + (end_x - start_x) * progress
                target_y = start_y + (end_y - start_y) * progress

                offset_x = random.randint(-20, 20) * intensity
                offset_y = random.randint(-10, 10) * intensity
                next_x = int(target_x + offset_x)
                next_y = int(target_y + offset_y)

                if lightning % 3 == 0:
                    color = theme['colors'][0]
                elif lightning % 3 == 1:
                    color = theme['colors'][1 % len(theme['colors'])]
                else:
                    color = theme['colors'][2 % len(theme['colors'])]

                line_width = max(1, int(1 + effective_energy * 4))

                draw.line([current_x, current_y, next_x, next_y], fill=color, width=line_width)

                for glow in range(1, 4):
                    glow_width = line_width + glow
                    r, g, b = hex_to_rgb(color)
                    glow_alpha = max(50, 200 // glow)
                    
                    h, s, v = colorsys.rgb_to_hsv(r / 255., g / 255., b / 255.)
                    v_glow = min(1.0, v * 1.5)
                    s_glow = min(1.0, s * 0.5)
                    r_glow, g_glow, b_glow = colorsys.hsv_to_rgb(h, s_glow, v_glow)
                    
                    simulated_r = int(r_glow * 255)
                    simulated_g = int(g_glow * 255)
                    simulated_b = int(b_glow * 255)
                    final_glow_color = (simulated_r, simulated_g, simulated_b, glow_alpha)
                    
                    draw.line([current_x, current_y, next_x, next_y], fill=final_glow_color, width=glow_width)

                current_x, current_y = next_x, next_y
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')

def create_barcode_visualizer(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                              theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float,
                              background_image: Optional[Image.Image], custom_text: str, text_font: Any,
                              text_color: str, text_position: str) -> Image.Image:
    width, height = resolution
    
    if background_image:
        img = background_image.resize(resolution, Image.LANCZOS).copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    else:
        img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)
        onset_strength = features['onset_strength'][time_idx]

        stft_slice = features['stft_magnitude'][:, time_idx]
        if np.max(stft_slice) > 0:
            stft_slice_norm = stft_slice / np.max(stft_slice)
        else:
            stft_slice_norm = np.zeros_like(stft_slice)

        num_bars = 100
        bar_spacing_ratio = 0.1
        
        nominal_bar_total_width = width / num_bars
        bar_width = nominal_bar_total_width * (1 - bar_spacing_ratio)
        # bar_space = nominal_bar_total_width * bar_spacing_ratio # Non usato direttamente

        for i in range(num_bars):
            x_start = int(i * nominal_bar_total_width)
            x_end = int(x_start + bar_width)

            if x_start >= width: continue

            freq_slice_start_bin = int((i / num_bars) * len(stft_slice_norm))
            freq_slice_end_bin = int(((i + 1) / num_bars) * len(stft_slice_norm))
            
            if freq_slice_start_bin >= len(stft_slice_norm): continue
            
            bar_freq_magnitude = np.mean(stft_slice_norm[freq_slice_start_bin:freq_slice_end_bin]) if freq_slice_end_bin > freq_slice_start_bin else 0

            max_col_height = height * 0.9 * intensity
            current_col_height = int(max_col_height * effective_energy * (0.5 + 0.5 * bar_freq_magnitude))
            current_col_height = max(5, min(current_col_height, height - 10))

            colors = theme['colors']
            color_index = int((bar_freq_magnitude) * (len(colors) - 1))
            color_index = min(color_index, len(colors) - 1)
            bar_color = colors[color_index]
            
            num_vertical_segments = 15
            segment_unit_height = current_col_height / num_vertical_segments

            segment_draw_prob_base = 0.3
            segment_draw_prob = segment_draw_prob_base + (0.7 * onset_strength) + (0.3 * bar_freq_magnitude)
            segment_draw_prob = np.clip(segment_draw_prob, 0, 1)

            for seg_idx in range(num_vertical_segments):
                seg_y_base = (height - current_col_height) // 2 + seg_idx * segment_unit_height

                seg_height = max(1, int(segment_unit_height * (0.5 + 0.5 * random.random() * (1 + onset_strength))))
                seg_height = min(seg_height, current_col_height - (seg_idx * segment_unit_height))

                y_offset_random = random.uniform(-onset_strength * height * 0.02 * intensity, 
                                                 onset_strength * height * 0.02 * intensity)
                
                segment_y1 = int(seg_y_base + y_offset_random)
                segment_y2 = int(segment_y1 + seg_height)

                segment_y1 = max(0, min(segment_y1, height))
                segment_y2 = max(0, min(segment_y2, height))
                
                if random.random() < segment_draw_prob:
                    if segment_y2 > segment_y1 and bar_width > 0:
                        draw.rectangle([x_start, segment_y1, x_end, segment_y2], fill=bar_color)
    
    draw_text_on_frame(draw, resolution, custom_text, text_font, text_color, text_position) # DISEGNA TESTO
    return img.convert('RGB')


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converte colore hex in RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_artistic_visualization(features: Dict[str, Any], style: str, resolution: Tuple[int, int],
                                  theme: Dict[str, Any], fps: int, intensity: float, global_volume_offset: float,
                                  output_dir: str, background_image: Optional[Image.Image], # AGGIUNTO
                                  custom_text: str, text_font: Any, text_color: str, text_position: str) -> int: # AGGIUNTI
    """Genera visualizzazione artistica salvando direttamente su disco."""
    duration = features['duration']
    total_frames = int(duration * fps)
    actual_frames = min(total_frames, MAX_DURATION * fps)

    # Mappatura stili a funzioni
    style_functions = {
        "Particle System": create_particle_system,
        "Circular Spectrum": create_circular_spectrum,
        "3D Waveforms": create_3d_waveforms,
        "Fluid Dynamics": create_fluid_dynamics,
        "Geometric Patterns": create_geometric_patterns,
        "Neural Network": create_neural_network,
        "Galaxy Spiral": create_galaxy_spiral,
        "Lightning Storm": create_lightning_storm,
        "Barcode Visualizer": create_barcode_visualizer
    }

    style_func = style_functions.get(style, create_particle_system)
    progress_bar = st.progress(0)
    
    for frame_idx in range(actual_frames):
        try:
            # Passa i nuovi parametri a tutte le funzioni di stile
            frame = style_func(features, frame_idx, resolution, theme, intensity,
                               fps, # Ho aggiunto fps qui, era mancante in alcune chiamate!
                               global_volume_offset, background_image, # AGGIUNTO
                               custom_text, text_font, text_color, text_position) # AGGIUNTI
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            frame.save(frame_path, format='JPEG', quality=85)
            
            progress = (frame_idx + 1) / actual_frames
            progress_bar.progress(progress)
            
            del frame
            if frame_idx % 50 == 0:
                gc.collect()
                
        except Exception as e:
            st.error(f"Errore generazione frame {frame_idx}: {e}")
            break
            
    return frame_idx + 1

def create_video_ffmpeg_pipe(fps: int, output_path: str, audio_path: str, frame_dir: str, frame_count: int) -> bool:
    """Crea video usando FFmpeg con input da pipe (zero-copy)"""
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'image2pipe',
            '-framerate', str(fps),
            '-i', '-',
            '-i', audio_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        for i in range(frame_count):
            frame_path = os.path.join(frame_dir, f"frame_{i:06d}.jpg")
            with open(frame_path, 'rb') as f:
                process.stdin.write(f.read())
            os.unlink(frame_path)
            
        process.stdin.close()
        process.wait()
        
        return process.returncode == 0
        
    except Exception as e:
        st.error(f"Errore creazione video: {e}")
        return False

def main():
    """Applicazione principale."""
    st.set_page_config(
        page_title="SoundWave Visualizer - Artistic Edition",
        page_icon="🎵",
        layout="wide"
    )

    st.markdown("<h1>🎵 SoundWave Visualizer - Artistic Edition <span style='font-size: 0.5em;'>by Loop507</span></h1>", unsafe_allow_html=True)
    st.markdown("*Trasforma la tua musica in arte visiva*")

    if not check_ffmpeg():
        st.error("❌ FFmpeg non trovato. Installare FFmpeg per continuare.")
        st.stop()

    with st.sidebar:
        st.header("🎨 Configurazioni Artistiche")

        selected_style = st.selectbox(
            "Stile Visualizzazione",
            list(ARTISTIC_STYLES.keys()),
            format_func=lambda x: ARTISTIC_STYLES[x]
        )

        st.subheader("Colori Personalizzati")
        bg_color = st.color_picker("Colore Sfondo", value="#000015")
        
        color_low_freq = st.color_picker("Colore Basse Frequenze", value="#FF0080")
        color_mid_freq = st.color_picker("Colore Medie Frequenze", value="#00FF80")
        color_high_freq = st.color_picker("Colore Alte Frequenze", value="#8000FF")
        
        selected_theme_data = {
            "colors": [color_low_freq, color_mid_freq, color_high_freq],
            "background": bg_color,
            "style": "custom"
        }
        selected_theme_name = "Personalizzato"

        movement_intensity = st.selectbox(
            "Intensità Movimento",
            list(MOVEMENT_INTENSITY.keys())
        )

        global_volume_offset = st.slider(
            "Volume Generale (Offset)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Aggiusta l'impatto del volume generale del brano sulla visualizzazione. Valori più alti rendono le forme più grandi/reattive per lo stesso volume audio."
        )

        # NUOVO: Caricamento immagine di sfondo
        st.subheader("Immagine di Sfondo")
        uploaded_background_image = st.file_uploader( # spostato in sidebar direttamente
            "Carica Immagine di Sfondo (Opzionale)",
            type=['png', 'jpg', 'jpeg'],
            help="Questa immagine verrà utilizzata come sfondo per la visualizzazione."
        )

        background_image = None
        if uploaded_background_image:
            try:
                background_image = Image.open(uploaded_background_image).convert("RGBA")
            except Exception as e:
                st.error(f"Errore caricamento immagine di sfondo: {e}")

        # NUOVO: Opzioni Testo Personalizzato
        st.subheader("Testo Personalizzato")
        custom_text = st.text_input("Testo da visualizzare", "SoundWave Art")
        text_font_size = st.slider("Dimensione Testo", min_value=10, max_value=100, value=30, step=5)
        text_color = st.color_picker("Colore Testo", value="#FFFFFF")
        text_position = st.selectbox(
            "Posizione Testo",
            ["Top-Left", "Top-Center", "Top-Right", "Bottom-Left", "Bottom-Center", "Bottom-Right", "Center"]
        )
        
        text_font = ImageFont.load_default() # Inizializza con font predefinito
        font_path = "arial.ttf" # Nome del file del font, assicurati che sia nella stessa directory
        try:
            if os.path.exists(font_path):
                text_font = ImageFont.truetype(font_path, text_font_size)
            else:
                st.warning("Font 'arial.ttf' non trovato. Utilizzo font predefinito.")
        except Exception as e:
            st.warning(f"Errore caricamento font: {e}. Utilizzo font predefinito.")


        format_ratio = st.selectbox(
            "Formato Video",
            list(FORMAT_RESOLUTIONS.keys())
        )

        fps = st.selectbox("Frame Rate", FPS_OPTIONS, index=1)

        st.markdown("---")
        st.markdown("### 📋 Info Limiti")
        st.info(f"""
        **Durata max:** {MAX_DURATION//60} minuti
        **File max:** {MAX_FILE_SIZE//(1024*1024)} MB
        """)

    uploaded_file = st.file_uploader(
        "🎵 Carica il tuo file audio",
        type=['mp3', 'wav', 'flac', 'm4a', 'aac'],
        help="Formati supportati: MP3, WAV, FLAC, M4A, AAC"
    )

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

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Durata", f"{duration:.1f}s")
            with col2:
                st.metric("Sample Rate", f"{sr} Hz")
            with col3:
                st.metric("Risoluzione", f"{FORMAT_RESOLUTIONS[format_ratio][0]}x{FORMAT_RESOLUTIONS[format_ratio][1]}")

            with st.spinner("🧠 Generando features audio avanzate..."):
                features = generate_enhanced_audio_features(y, sr, fps)

            if features is None:
                st.error("Errore nell'analisi audio.")
                st.stop()

            st.markdown("### 📊 Analisi Frequenze in Percentuali")
            avg_bass = np.mean(features['freq_bass']) * 100 if features['freq_bass'].size > 0 else 0
            avg_mid = np.mean(features['freq_high_mid']) * 100 if features['freq_high_mid'].size > 0 else 0
            avg_high = np.mean(features['freq_brilliance']) * 100 if features['freq_brilliance'].size > 0 else 0

            bass_hz_range = features['bass_hz_range']
            mid_hz_range = features['mid_hz_range']
            high_hz_range = features['high_hz_range']

            freq_col1, freq_col2, freq_col3 = st.columns(3)
            with freq_col1:
                st.metric(
                    "Basse",
                    f"{avg_bass:.1f}%",
                    help=f"Range: {bass_hz_range[0]:.0f}Hz - {bass_hz_range[1]:.0f}Hz"
                )
            with freq_col2:
                st.metric(
                    "Medie",
                    f"{avg_mid:.1f}%",
                    help=f"Range: {mid_hz_range[0]:.0f}Hz - {mid_hz_range[1]:.0f}Hz"
                )
            with freq_col3:
                st.metric(
                    "Alte",
                    f"{avg_high:.1f}%",
                    help=f"Range: {high_hz_range[0]:.0f}Hz - {high_hz_range[1]:.0f}Hz"
                )
            st.info("""
            Questi valori indicano la presenza media di ciascuna banda di frequenza nel brano (0-100%).
            I range in Hz si riferiscono alle specifiche suddivisioni interne utilizzate per la visualizzazione, ora più vicine a quelle musicali.
            """)


            st.markdown("### 🎨 Anteprima Configurazione")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **Stile:** {ARTISTIC_STYLES[selected_style]}
                **Tema:** {selected_theme_name}
                **Intensità:** {movement_intensity}
                """)

            with col2:
                st.info(f"""
                **Formato:** {format_ratio}
                **FPS:** {fps}
                **Volume Offset:** {global_volume_offset}
                **Frames totali:** ~{int(duration * fps)}
                """)

            if st.button("🚀 Genera Visualizzazione Artistica", type="primary"):
                intensity_value = MOVEMENT_INTENSITY[movement_intensity]
                resolution = FORMAT_RESOLUTIONS[format_ratio]
                
                with tempfile.TemporaryDirectory() as frame_dir:
                    with st.spinner("🎨 Creando arte visiva..."):
                        frame_count = generate_artistic_visualization(
                            features, selected_style, resolution, selected_theme_data, fps,
                            intensity_value, global_volume_offset, frame_dir, background_image, # AGGIUNTO
                            custom_text, text_font, text_color, text_position # AGGIUNTI
                        )
                    
                    if frame_count > 0:
                        output_path = f"soundwave_artistic_{selected_style.lower().replace(' ', '_')}.mp4"
                        
                        with st.spinner("🎬 Creando video finale..."):
                            success = create_video_ffmpeg_pipe(
                                fps, output_path, temp_audio_path, frame_dir, frame_count
                            )
                        
                        if success and os.path.exists(output_path):
                            st.success("✅ Video creato con successo!")

                            st.video(output_path)

                            with open(output_path, 'rb') as video_file:
                                st.download_button(
                                    "📥 Scarica Video",
                                    video_file.read(),
                                    file_name=output_path,
                                    mime="video/mp4"
                                )
                        else:
                            st.error("❌ Errore nella creazione del video.")
                    else:
                        st.error("❌ Nessun frame generato.")

        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

if __name__ == "__main__":
    main()
