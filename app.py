# app.py - SoundWave Visualizer by Loop507 - ARTISTIC EDITION
import streamlit as st
import numpy as np
import librosa
import os
import subprocess
import gc
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFilter
import io
from typing import Tuple, Optional, Dict, Any
import colorsys
import math
import random

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

# NUOVI STILI ARTISTICI
ARTISTIC_STYLES: Dict[str, str] = {
    "Particle System": "🌟 Particelle che danzano con la musica",
    "Circular Spectrum": "⭕ Spettro radiale rotante",
    "3D Waveforms": "🌊 Onde pseudo-3D dinamiche",
    "Fluid Dynamics": "💧 Simulazione fluidi realistici",
    "Geometric Patterns": "🔷 Forme geometriche animate",
    "Neural Network": "🧠 Rete neurale pulsante",
    "Galaxy Spiral": "🌌 Spirale galattica in movimento",
    "Lightning Storm": "⚡ Tempesta elettrica musicale"
}

# INTENSITÀ MOVIMENTO
MOVEMENT_INTENSITY: Dict[str, float] = {
    "Soft": 0.3,
    "Medio": 0.7,
    "Hard": 1.2
}

# TEMI COLORE ARTISTICI
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

        # Analisi dettagliata delle frequenze (più bande)
        n_freqs = stft_norm.shape[0]
        freq_sub_bass = stft_norm[:n_freqs//8, :]
        freq_bass = stft_norm[n_freqs//8:n_freqs//4, :]
        freq_low_mid = stft_norm[n_freqs//4:n_freqs//2, :]
        freq_high_mid = stft_norm[n_freqs//2:3*n_freqs//4, :]
        freq_presence = stft_norm[3*n_freqs//4:7*n_freqs//8, :]
        freq_brilliance = stft_norm[7*n_freqs//8:, :]

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
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

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

            # Bande frequenza dettagliate
            'freq_sub_bass': freq_sub_bass,
            'freq_bass': freq_bass,
            'freq_low_mid': freq_low_mid,
            'freq_high_mid': freq_high_mid,
            'freq_presence': freq_presence,
            'freq_brilliance': freq_brilliance,

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
            'magnitude_raw': magnitude
        }
    except Exception as e:
        st.error(f"Errore feature avanzate: {e}")
        return None

def create_particle_system(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Crea sistema particellare che danza con la musica."""
    width, height = resolution
    img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    draw = ImageDraw.Draw(img)

    # Calcola indice temporale
    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['rms_energy'].shape[0] - 1)

    if time_idx >= 0:
        # Energia attuale e vicine per movimento fluido
        current_energy = features['rms_energy'][time_idx]
        onset_strength = features['onset_strength'][time_idx] if time_idx < len(features['onset_strength']) else 0

        # Ottieni valori spettrali per colori
        bass_energy = np.mean(features['freq_bass'][:, time_idx]) if time_idx < features['freq_bass'].shape[1] else 0
        mid_energy = np.mean(features['freq_high_mid'][:, time_idx]) if time_idx < features['freq_high_mid'].shape[1] else 0
        high_energy = np.mean(features['freq_brilliance'][:, time_idx]) if time_idx < features['freq_brilliance'].shape[1] else 0

        # Numero particelle basato su energia
        num_particles = int(50 + current_energy * 200 * intensity)

        # Genera particelle
        for i in range(num_particles):
            # Posizione influenzata da frequenze diverse
            angle = (i / num_particles) * 2 * np.pi + frame_idx * 0.05 * intensity

            # Raggio basato su energia e frequenze
            base_radius = min(width, height) * 0.1
            radius_variation = (bass_energy * 0.4 + mid_energy * 0.3 + high_energy * 0.3) * min(width, height) * 0.3
            radius = base_radius + radius_variation * np.sin(angle * 3 + frame_idx * 0.1)

            center_x, center_y = width // 2, height // 2
            x = int(center_x + radius * np.cos(angle) * intensity)
            y = int(center_y + radius * np.sin(angle) * intensity)

            # Dimensione particella
            particle_size = int(2 + onset_strength * 15 + current_energy * 10)

            # Colore basato su frequenza dominante
            if bass_energy > mid_energy and bass_energy > high_energy:
                color = theme['colors'][0]  # Bassi
            elif mid_energy > high_energy:
                color = theme['colors'][1]  # Medi
            else:
                color = theme['colors'][2]  # Acuti

            # Trasparenza basata su energia
            alpha = int(100 + current_energy * 155)

            # Disegna particella con glow
            if theme['style'] == 'neon':
                # Effetto glow per neon
                for glow_radius in range(particle_size + 4, particle_size - 1, -1):
                    glow_alpha = max(10, alpha // (glow_radius - particle_size + 2))
                    glow_color = (*hex_to_rgb(color), glow_alpha)
                    draw.ellipse([x - glow_radius, y - glow_radius,
                                x + glow_radius, y + glow_radius], fill=glow_color)
            else:
                # Particella normale
                draw.ellipse([x - particle_size, y - particle_size,
                            x + particle_size, y + particle_size],
                           fill=(*hex_to_rgb(color), alpha))

    return img.convert('RGB')

def create_circular_spectrum(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                           theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Crea spettro circolare rotante."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['stft_magnitude'].shape[1] - 1)

    if time_idx >= 0:
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) * 0.4

        # Ottieni dati spettrali
        spectrum_slice = features['stft_magnitude'][:, time_idx]
        n_bins = len(spectrum_slice)

        # Rotazione basata su tempo e intensità
        rotation_offset = frame_idx * 0.02 * intensity

        for i in range(0, n_bins, max(1, n_bins // 120)):  # Campiona spettro
            if i < len(spectrum_slice):
                magnitude = spectrum_slice[i]

                # Angolo
                angle = (i / n_bins) * 2 * np.pi + rotation_offset

                # Raggio interno e esterno
                inner_radius = max_radius * 0.3
                outer_radius = inner_radius + magnitude * max_radius * 0.6 * intensity

                # Posizioni
                x1 = int(center_x + inner_radius * np.cos(angle))
                y1 = int(center_y + inner_radius * np.sin(angle))
                x2 = int(center_x + outer_radius * np.cos(angle))
                y2 = int(center_y + outer_radius * np.sin(angle))

                # Colore basato su frequenza
                color_idx = int((i / n_bins) * len(theme['colors']))
                color_idx = min(color_idx, len(theme['colors']) - 1)
                color = theme['colors'][color_idx]

                # Spessore linea basato su magnitudine
                line_width = max(1, int(magnitude * 5 + 1))

                # Disegna linea radiale
                draw.line([x1, y1, x2, y2], fill=color, width=line_width)

    return img

def create_3d_waveforms(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                       theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Crea onde pseudo-3D dinamiche."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['rms_energy'].shape[0] - 1)

    if time_idx >= 0:
        # Finestra temporale per waveform
        window_size = min(100, features['rms_energy'].shape[0] // 4)
        start_idx = max(0, time_idx - window_size // 2)
        end_idx = min(features['rms_energy'].shape[0], start_idx + window_size)

        if end_idx > start_idx:
            waveform_data = features['rms_energy'][start_idx:end_idx]

            # Crea multiple "layers" per effetto 3D
            layers = 5
            for layer in range(layers):
                layer_offset_y = layer * 20
                layer_alpha = 255 - layer * 30

                # Colore layer
                base_color = theme['colors'][layer % len(theme['colors'])]

                points = []
                for i, amplitude in enumerate(waveform_data):
                    x = int((i / len(waveform_data)) * width)

                    # Effetto 3D: oscillazione verticale + prospettiva
                    base_y = height // 2 + layer_offset_y
                    wave_y = int(amplitude * height * 0.3 * intensity * np.sin(frame_idx * 0.1 + layer))
                    y = base_y - wave_y

                    points.append((x, y))

                # Disegna layer
                if len(points) > 1:
                    # Crea poligono riempito per effetto volume
                    fill_points = points + [(width, height), (0, height)]

                    # Colore con trasparenza
                    r, g, b = hex_to_rgb(base_color)
                    layer_color = f"#{r:02x}{g:02x}{b:02x}"

                    # Disegna riempimento
                    draw.polygon(fill_points, fill=layer_color)

                    # Linea di contorno
                    for i in range(len(points) - 1):
                        draw.line([points[i], points[i + 1]], fill=layer_color, width=2)

    return img

def create_fluid_dynamics(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Simula dinamica fluidi con la musica."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['rms_energy'].shape[0] - 1)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]

        # Parametri fluido
        num_waves = 8

        for wave_idx in range(num_waves):
            # Fase della singola onda
            wave_phase = frame_idx * 0.05 * intensity + wave_idx * np.pi / 4

            # Ampiezza basata su energia
            amplitude = current_energy * height * 0.2 * intensity

            # Frequenza onda
            frequency = 0.01 + wave_idx * 0.005

            points = []
            for x in range(0, width, 5):  # Ogni 5 pixel per performance
                # Onda sinusoidale con variazioni
                y_base = height // 2 + wave_idx * 30 - num_waves * 15
                y_offset = amplitude * np.sin(x * frequency + wave_phase)
                y = int(y_base + y_offset)

                # Mantieni nel range
                y = max(0, min(height - 1, y))
                points.append((x, y))

            # Colore onda
            color_idx = wave_idx % len(theme['colors'])
            color = theme['colors'][color_idx]

            # Disegna onda come poligono riempito
            if len(points) > 2:
                # Crea forma chiusa per riempimento
                bottom_points = [(width, height), (0, height)]
                wave_polygon = points + bottom_points

                # Trasparenza per sovrapposizione
                r, g, b = hex_to_rgb(color)
                alpha = int(100 + current_energy * 100)

                # PIL non supporta alpha direttamente, usiamo colori più scuri per trasparenza
                dark_factor = alpha / 255
                r = int(r * dark_factor)
                g = int(g * dark_factor)
                b = int(b * dark_factor)
                wave_color = f"#{r:02x}{g:02x}{b:02x}"

                draw.polygon(wave_polygon, fill=wave_color)

                # Contorno
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i + 1]], fill=color, width=2)

    return img

def create_geometric_patterns(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                            theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Crea pattern geometrici animati."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['rms_energy'].shape[0] - 1)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        center_x, center_y = width // 2, height // 2

        # Pattern concentrici
        num_rings = 6
        base_radius = min(width, height) * 0.05

        for ring in range(num_rings):
            # Raggio anello
            radius = base_radius + ring * (min(width, height) * 0.08)
            radius *= (1 + current_energy * intensity * 0.5)  # Pulsazione

            # Rotazione
            rotation = frame_idx * 0.02 * intensity + ring * 0.5

            # Numero lati poligono
            sides = 3 + ring

            # Punti poligono
            points = []
            for side in range(sides):
                angle = (side / sides) * 2 * np.pi + rotation
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append((x, y))

            # Colore
            color = theme['colors'][ring % len(theme['colors'])]

            # Disegna poligono
            if len(points) > 2:
                if ring % 2 == 0:
                    # Riempito
                    draw.polygon(points, fill=color)
                else:
                    # Solo contorno
                    draw.polygon(points, outline=color, width=3)

    return img

def create_neural_network(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Crea visualizzazione rete neurale pulsante."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['rms_energy'].shape[0] - 1)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]

        # Nodi della rete
        nodes = []
        grid_size = 8

        for i in range(grid_size):
            for j in range(grid_size):
                x = int((i + 1) / (grid_size + 1) * width)
                y = int((j + 1) / (grid_size + 1) * height)

                # Attivazione nodo basata su energia e posizione
                activation = current_energy + 0.3 * np.sin(frame_idx * 0.1 + i + j)
                activation = max(0, min(1, activation))

                nodes.append((x, y, activation))

        # Connessioni tra nodi vicini
        for i, (x1, y1, act1) in enumerate(nodes):
            for j, (x2, y2, act2) in enumerate(nodes[i+1:], i+1):
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                # Connetti solo nodi vicini
                if distance < min(width, height) * 0.2:
                    # Intensità connessione
                    connection_strength = (act1 + act2) / 2 * intensity

                    if connection_strength > 0.3:
                        # Spessore linea
                        line_width = max(1, int(connection_strength * 3))

                        # Colore connessione
                        color_idx = int(connection_strength * len(theme['colors']))
                        color_idx = min(color_idx, len(theme['colors']) - 1)
                        color = theme['colors'][color_idx]

                        draw.line([x1, y1, x2, y2], fill=color, width=line_width)

        # Disegna nodi
        for x, y, activation in nodes:
            if activation > 0.2:
                node_size = int(3 + activation * 8 * intensity)
                color_idx = int(activation * len(theme['colors']))
                color_idx = min(color_idx, len(theme['colors']) - 1)
                color = theme['colors'][color_idx]

                draw.ellipse([x-node_size, y-node_size, x+node_size, y+node_size], fill=color)

    return img

def create_galaxy_spiral(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                        theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Crea spirale galattica in movimento."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['rms_energy'].shape[0] - 1)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        center_x, center_y = width // 2, height // 2

        # Parametri spirale
        num_arms = 3
        points_per_arm = 100

        for arm in range(num_arms):
            arm_offset = (arm / num_arms) * 2 * np.pi

            points = []
            for i in range(points_per_arm):
                # Parametro spirale
                t = i / points_per_arm * 6 * np.pi  # 3 giri

                # Raggio crescente
                radius = (i / points_per_arm) * min(width, height) * 0.4
                radius *= (1 + current_energy * intensity * 0.3)

                # Angolo con rotazione
                angle = t + arm_offset + frame_idx * 0.02 * intensity

                # Posizione
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))

                points.append((x, y))

            # Colore braccio spirale
            color = theme['colors'][arm % len(theme['colors'])]

            # Disegna spirale come linea continua
            if len(points) > 1:
                for i in range(len(points) - 1):
                    # Intensità diminuisce verso l'esterno
                    intensity_factor = 1 - (i / len(points))
                    line_width = max(1, int(2 + current_energy * 3 * intensity_factor))
                    draw.line([points[i], points[i + 1]], fill=color, width=line_width)

            # Stelle lungo la spirale
            star_density = int(10 + current_energy * 20)
            for star in range(star_density):
                point_idx = random.randint(0, len(points) - 1)
                if point_idx < len(points):
                    star_x, star_y = points[point_idx]
                    # Piccola perturbazione
                    star_x += random.randint(-5, 5)
                    star_y += random.randint(-5, 5)

                    star_size = random.randint(1, 3)
                    draw.ellipse([star_x-star_size, star_y-star_size,
                                star_x+star_size, star_y+star_size], fill=color)

    return img

def create_lightning_storm(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                          theme: Dict[str, Any], intensity: float, fps: int) -> Image.Image:
    """Crea tempesta elettrica musicale."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = int((frame_idx / fps) * features['sr'] / features['hop_length'])
    time_idx = min(time_idx, features['rms_energy'].shape[0] - 1)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        onset_strength = features['onset_strength'][time_idx] if time_idx < len(features['onset_strength']) else 0

        # Numero fulmini basato su energia
        num_lightning = int(1 + onset_strength * 8 + current_energy * 5)

        for lightning in range(num_lightning):
            # Punto di partenza (alto)
            start_x = random.randint(0, width)
            start_y = random.randint(0, height // 4)

            # Punto di arrivo (basso)
            end_x = start_x + random.randint(-width//4, width//4)
            end_y = random.randint(3*height//4, height)

            # Segmenti fulmine
            current_x, current_y = start_x, start_y

            segments = 8 + int(current_energy * 12)

            for segment in range(segments):
                # Punto successivo
                progress = (segment + 1) / segments

                target_x = start_x + (end_x - start_x) * progress
                target_y = start_y + (end_y - start_y) * progress

                # Perturbazione casuale
                offset_x = random.randint(-20, 20) * intensity
                offset_y = random.randint(-10, 10) * intensity
                next_x = int(target_x + offset_x)
                next_y = int(target_y + offset_y)

                # Colore fulmine
                color = theme['colors'][lightning % len(theme['colors'])]

                # Spessore basato su intensità
                line_width = max(1, int(1 + current_energy * 4))

                # Disegna segmento
                draw.line([current_x, current_y, next_x, next_y], fill=color, width=line_width)

                # Effetto glow
                if theme['style'] == 'neon':
                    for glow in range(1, 4):
                        glow_width = line_width + glow
                        r, g, b = hex_to_rgb(color)
                        glow_alpha = max(50, 200 // glow)
                        # Simula glow con colori più chiari
                        glow_r = min(255, r + glow * 20)
                        glow_g = min(255, g + glow * 20)
                        glow_b = min(255, b + glow * 20)
                        glow_color = f"#{glow_r:02x}{glow_g:02x}{glow_b:02x}"
                        draw.line([current_x, current_y, next_x, next_y], fill=glow_color, width=glow_width)

                current_x, current_y = next_x, next_y

    return img

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converte colore hex in RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_artistic_visualization(features: Dict[str, Any], style: str, resolution: Tuple[int, int],
                                  theme: Dict[str, Any], fps: int, intensity: float) -> list:
    """Genera visualizzazione artistica frame per frame."""
    duration = features['duration']
    total_frames = int(duration * fps)
    frames = []

    # Mappatura stili a funzioni
    style_functions = {
        "Particle System": create_particle_system,
        "Circular Spectrum": create_circular_spectrum,
        "3D Waveforms": create_3d_waveforms,
        "Fluid Dynamics": create_fluid_dynamics,
        "Geometric Patterns": create_geometric_patterns,
        "Neural Network": create_neural_network,
        "Galaxy Spiral": create_galaxy_spiral,
        "Lightning Storm": create_lightning_storm
    }

    style_func = style_functions.get(style, create_particle_system)

    progress_bar = st.progress(0)
    for frame_idx in range(min(total_frames, MAX_DURATION * fps)):  # Limite sicurezza
        try:
            frame = style_func(features, frame_idx, resolution, theme, intensity, fps)
            frames.append(frame)

            # Aggiorna progress bar
            progress = (frame_idx + 1) / min(total_frames, MAX_DURATION * fps)
            progress_bar.progress(progress)

            # Garbage collection periodico
            if frame_idx % 50 == 0:
                gc.collect()

        except Exception as e:
            st.error(f"Errore generazione frame {frame_idx}: {e}")
            break

    return frames

def create_video_from_frames(frames: list, fps: int, output_path: str, audio_path: str) -> bool:
    """Crea video da frames con audio."""
    if not frames:
        st.error("Nessun frame da processare.")
        return False

    try:
        # Salva frames temporanei
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            frame.save(frame_path)
            frame_paths.append(frame_path)

        # Crea video temporaneo senza audio
        temp_video = os.path.join(temp_dir, "temp_video.mp4")

        # Comando FFmpeg per creare video da immagini
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # Qualità alta
            temp_video
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Errore FFmpeg video: {result.stderr}")
            return False

        # Aggiungi audio
        final_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]

        result = subprocess.run(final_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Errore FFmpeg audio: {result.stderr}")
            return False

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        return True

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

    st.title("🎵 SoundWave Visualizer - Artistic Edition")
    st.markdown("*Trasforma la tua musica in arte visiva*")

    # Check FFmpeg
    if not check_ffmpeg():
        st.error("❌ FFmpeg non trovato. Installare FFmpeg per continuare.")
        st.stop()

    # Sidebar configurazioni
    with st.sidebar:
        st.header("🎨 Configurazioni Artistiche")

        # Stile artistico
        selected_style = st.selectbox(
            "Stile Visualizzazione",
            list(ARTISTIC_STYLES.keys()),
            format_func=lambda x: ARTISTIC_STYLES[x]
        )

        # Tema colori
        selected_theme = st.selectbox(
            "Tema Colori",
            list(ARTISTIC_COLOR_THEMES.keys())
        )

        # Intensità movimento
        movement_intensity = st.selectbox(
            "Intensità Movimento",
            list(MOVEMENT_INTENSITY.keys())
        )

        # Risoluzione
        format_ratio = st.selectbox(
            "Formato Video",
            list(FORMAT_RESOLUTIONS.keys())
        )

        # FPS
        fps = st.selectbox("Frame Rate", FPS_OPTIONS, index=1)

        st.markdown("---")
        st.markdown("### 📋 Info Limiti")
        st.info(f"""
        **Durata max:** {MAX_DURATION//60} minuti
        **File max:** {MAX_FILE_SIZE//(1024*1024)} MB
        """)

    # Upload file
    uploaded_file = st.file_uploader(
        "🎵 Carica il tuo file audio",
        type=['mp3', 'wav', 'flac', 'm4a', 'aac'],
        help="Formati supportati: MP3, WAV, FLAC, M4A, AAC"
    )

    if uploaded_file:
        if not validate_audio_file(uploaded_file):
            st.stop()

        # Salva file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name

        try:
            # Carica e processa audio
            with st.spinner("🎵 Analizzando audio..."):
                y, sr, duration = load_and_process_audio(temp_audio_path)

            if y is None:
                st.error("Impossibile caricare il file audio.")
                st.stop()

            # Mostra info audio
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Durata", f"{duration:.1f}s")
            with col2:
                st.metric("Sample Rate", f"{sr} Hz")
            with col3:
                st.metric("Risoluzione", f"{FORMAT_RESOLUTIONS[format_ratio][0]}x{FORMAT_RESOLUTIONS[format_ratio][1]}")

            # Genera features avanzate
            with st.spinner("🧠 Generando features audio avanzate..."):
                features = generate_enhanced_audio_features(y, sr, fps)

            if features is None:
                st.error("Errore nell'analisi audio.")
                st.stop()

            # Anteprima configurazione
            st.markdown("### 🎨 Anteprima Configurazione")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **Stile:** {ARTISTIC_STYLES[selected_style]}
                **Tema:** {selected_theme}
                **Intensità:** {movement_intensity}
                """)

            with col2:
                st.info(f"""
                **Formato:** {format_ratio}
                **FPS:** {fps}
                **Frames totali:** ~{int(duration * fps)}
                """)

            # Bottone genera
            if st.button("🚀 Genera Visualizzazione Artistica", type="primary"):
                theme = ARTISTIC_COLOR_THEMES[selected_theme]
                intensity = MOVEMENT_INTENSITY[movement_intensity]
                resolution = FORMAT_RESOLUTIONS[format_ratio]

                # Genera visualizzazione
                with st.spinner("🎨 Creando arte visiva..."):
                    frames = generate_artistic_visualization(
                        features, selected_style, resolution, theme, fps, intensity
                    )

                if frames:
                    # Crea video
                    output_path = f"soundwave_artistic_{selected_style.lower().replace(' ', '_')}.mp4"

                    with st.spinner("🎬 Creando video finale..."):
                        success = create_video_from_frames(frames, fps, output_path, temp_audio_path)

                    if success and os.path.exists(output_path):
                        st.success("✅ Video creato con successo!")

                        # Mostra video
                        st.video(output_path)

                        # Download
                        with open(output_path, 'rb') as video_file:
                            st.download_button(
                                "📥 Scarica Video",
                                video_file.read(),
                                file_name=output_path,
                                mime="video/mp4"
                            )
                    else:
                        st.error("❌ Errore nella creazione del video.")

        finally:
            # Cleanup
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

if __name__ == "__main__":
    main()
