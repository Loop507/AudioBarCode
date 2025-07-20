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
from PIL import Image, ImageDraw, ImageColor, ImageFilter
import io
from typing import Tuple, Optional, Dict, Any
import colorsys
import math
import random
import sys
# import importlib # Non più necessario se non gestisci import dinamici dopo la rimozione

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
    "Soft": 0.2, # Ridotto per una differenza più marcata
    "Medio": 1.0, # Aumentato per una chiara distinzione
    "Hard": 3.0 # Aumentato significativamente per un effetto molto pronunciato
}

# TEMI COLORE ARTISTICI - MANTENUTI PER GLI ALTRI STILI, MA USEREMO SCELTA PERSONALIZZATA PER PARTICELLE
# Questi temi non saranno più selezionabili direttamente, ma i loro colori predefiniti possono essere usati come esempio per i color picker.
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
# Questi sono range tipici per l'analisi musicale
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
        # BASS_HZ_RANGE
        bass_start_bin = np.searchsorted(freq_bins_hz, BASS_HZ_RANGE[0])
        bass_end_bin = np.searchsorted(freq_bins_hz, BASS_HZ_RANGE[1], side='right') - 1
        # Assicurati che gli indici siano validi e in ordine
        bass_start_bin = max(0, bass_start_bin)
        bass_end_bin = min(len(freq_bins_hz) - 1, bass_end_bin)
        freq_bass = stft_norm[bass_start_bin : bass_end_bin + 1, :]

        # MID_HZ_RANGE
        mid_start_bin = np.searchsorted(freq_bins_hz, MID_HZ_RANGE[0])
        mid_end_bin = np.searchsorted(freq_bins_hz, MID_HZ_RANGE[1], side='right') - 1
        mid_start_bin = max(0, mid_start_bin)
        mid_end_bin = min(len(freq_bins_hz) - 1, mid_end_bin)
        freq_high_mid = stft_norm[mid_start_bin : mid_end_bin + 1, :]

        # HIGH_HZ_RANGE
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
        # CORREZIONE QUI: Usare spectral_rolloff.min() e spectral_rolloff.max()
        rolloff_norm = (spectral_rolloff - spectral_rolloff.min()) / (spectral_rolloff.max() - spectral_rolloff.min() + 1e-9)
        bandwidth_norm = (spectral_bandwidth - spectral_bandwidth.min()) / (spectral_bandwidth.max() - spectral_bandwidth.min() + 1e-9)
        zcr_norm = (zero_crossing_rate - zero_crossing_rate.min()) / (zero_crossing_rate.max() - zero_crossing_rate.min() + 1e-9)

        # RMS Energy e onset detection
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        # rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9) # Vecchia normalizzazione
        rms_norm = np.clip(rms / np.max(rms) if np.max(rms) > 0 else rms, 0, 1) # Nuova normalizzazione: RMS relativo al suo picco nel brano, quindi sensibile al volume relativo ma ancora 0-1

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
            'rms_energy': rms_norm, # Questa è la feature che verrà scalata dal nuovo slider
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
            'n_fft': n_fft # Aggiungi n_fft per riferimento a freq_bins
        }
    except Exception as e:
        st.error(f"Errore feature avanzate: {e}")
        return None

def get_time_idx(features: Dict[str, Any], frame_idx: int, fps: int) -> int:
    """Calcola l'indice temporale STFT/feature per un dato frame video."""
    current_time = frame_idx / fps
    time_idx = librosa.time_to_frames(current_time, sr=features['sr'], hop_length=features['hop_length'])
    # Assicurati che l'indice non superi i limiti dell'array delle feature
    return min(time_idx, features['rms_energy'].shape[0] - 1)


def create_particle_system(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Crea sistema particellare che danza con la musica (Modificato per maggiore intensità)."""
    width, height = resolution
    img = Image.new('RGBA', (width, height), (*hex_to_rgb(theme['background']), 255))
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        # Energia attuale e vicine per movimento fluido
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1) # Applica offset e clippa
        onset_strength = features['onset_strength'][time_idx] # onsets are also indexed by time_idx

        # Ottieni valori spettrali per colori (questi non sono direttamente influenzati dal volume generale per mantenere coerenza cromatica)
        bass_energy = np.mean(features['freq_bass'][:, time_idx])
        mid_energy = np.mean(features['freq_high_mid'][:, time_idx])
        high_energy = np.mean(features['freq_brilliance'][:, time_idx])

        # NUOVE IMPOSTAZIONI PER PARTICELLE PIÙ INTENSE E PIENE
        # Numero particelle aumentato e più reattivo
        num_particles = int(200 + effective_energy * 500 * intensity) # Aumentato base e reattività

        # Genera particelle
        for i in range(num_particles):
            # Posizione influenzata da frequenze diverse
            angle = (i / num_particles) * 2 * np.pi + frame_idx * 0.08 * intensity # Velocità di rotazione aumentata

            # Raggio basato su energia e frequenze
            base_radius = min(width, height) * 0.15 # Raggio base leggermente più grande
            radius_variation = (bass_energy * 0.5 + mid_energy * 0.3 + high_energy * 0.2) * min(width, height) * 0.4 # Più variazione
            radius = base_radius + radius_variation * (1 + np.sin(angle * 5 + frame_idx * 0.15)) # Oscillazione più complessa e veloce

            center_x, center_y = width // 2, height // 2
            x = int(center_x + radius * np.cos(angle) * (1 + effective_energy * 0.2)) # Posizione più reattiva
            y = int(center_y + radius * np.sin(angle) * (1 + effective_energy * 0.2))

            # Dimensione particella più grande e reattiva
            particle_size = int(4 + onset_strength * 25 + effective_energy * 15 * intensity) # Dimensione base aumentata e reattività
            particle_size = max(1, particle_size) # Assicura dimensione minima

            # Colore basato su frequenza dominante o ciclico se ci sono più colori
            colors = theme['colors']
            if len(colors) == 1:
                color = colors[0]
            else:
                # Assegna i colori in base alla dominanza delle frequenze
                if bass_energy > mid_energy and bass_energy > high_energy:
                    color = colors[0]  # Corrisponde a "Colore Basse Frequenze"
                elif mid_energy > high_energy:
                    color = colors[1 % len(colors)]  # Corrisponde a "Colore Medie Frequenze"
                else:
                    color = colors[2 % len(colors)]  # Corrisponde a "Colore Alte Frequenze"
            
            # Trasparenza basata su energia
            alpha = int(150 + effective_energy * 105) # Trasparenza base più alta

            # Disegna particella con glow potenziato
            for glow_radius in range(particle_size + 6, particle_size - 1, -1): # Range glow più ampio
                glow_alpha = max(10, alpha // (glow_radius - particle_size + 2)) # Decay più lento del glow
                if glow_radius > particle_size: # Per il glow esterno
                    r, g, b = hex_to_rgb(color)
                    h, s, v = colorsys.rgb_to_hsv(r / 255., g / 255., b / 255.)
                    v_glow = min(1.0, v * 1.5) # Aumenta la luminosità
                    s_glow = min(1.0, s * 0.8) # Mantiene saturazione, ma può essere ridotta per un glow più diffuso
                    r_glow, g_glow, b_glow = colorsys.hsv_to_rgb(h, s_glow, v_glow)
                    r_glow_int = int(r_glow * 255)
                    g_glow_int = int(g_glow * 255)
                    b_glow_int = int(b_glow * 255)
                    final_glow_color = (r_glow_int, g_glow_int, b_glow_int, glow_alpha)
                else: # Per la particella centrale
                    final_glow_color = (*hex_to_rgb(color), alpha) # Particella interna opaca

                draw.ellipse([x - glow_radius, y - glow_radius,
                            x + glow_radius, y + glow_radius], fill=final_glow_color)

    return img.convert('RGB')


def create_circular_spectrum(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                           theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Crea spettro circolare rotante."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) * 0.4

        # Ottieni dati spettrali
        spectrum_slice = features['stft_magnitude'][:, time_idx]
        n_bins = len(spectrum_slice)

        # Rotazione basata su tempo e intensità
        rotation_offset = frame_idx * 0.02 * intensity

        # Energia attuale per scalare la reattività del raggio
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        for i in range(0, n_bins, max(1, n_bins // 120)):  # Campiona spettro
            if i < len(spectrum_slice):
                magnitude = spectrum_slice[i]

                # Angolo
                angle = (i / n_bins) * 2 * np.pi + rotation_offset

                # Raggio interno e esterno, influenzati dall'energia effettiva
                inner_radius = max_radius * 0.3
                outer_radius = inner_radius + magnitude * max_radius * 0.6 * intensity * effective_energy # Scalato dall'offset

                # Posizioni
                x1 = int(center_x + inner_radius * np.cos(angle))
                y1 = int(center_y + inner_radius * np.sin(angle))
                x2 = int(center_x + outer_radius * np.cos(angle))
                y2 = int(center_y + outer_radius * np.sin(angle))

                # Colore basato su frequenza usando i colori personalizzati
                colors = theme['colors']
                if len(colors) == 1:
                    color = colors[0]
                else:
                    # Distribuisci i colori sui bin dello spettro, ad es. i primi colori per le basse, gli ultimi per le alte
                    # Assumiamo che colors[0] sia per basse, colors[1] per medie, colors[2] per alte
                    if i < n_bins // 3: # Basse
                        color = colors[0 % len(colors)]
                    elif i < 2 * n_bins // 3: # Medie
                        color = colors[1 % len(colors)]
                    else: # Alte
                        color = colors[2 % len(colors)]


                # Spessore linea basato su magnitudine e energia effettiva
                line_width = max(1, int(magnitude * 5 * effective_energy + 1)) # Scalato dall'offset

                # Disegna linea radiale
                draw.line([x1, y1, x2, y2], fill=color, width=line_width)

    return img

def create_3d_waveforms(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                       theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Crea onde pseudo-3D dinamiche."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        # Finestra temporale per waveform
        window_size = min(100, features['rms_energy'].shape[0] // 4)
        start_idx = max(0, time_idx - window_size // 2)
        end_idx = min(features['rms_energy'].shape[0], start_idx + window_size)

        if end_idx > start_idx:
            # Qui usiamo la waveform originale normalizzata, ma la sua altezza sarà influenzata da effective_energy
            waveform_data = features['rms_energy'][start_idx:end_idx]

            # Crea multiple "layers" per effetto 3D
            layers = 5
            for layer in range(layers):
                layer_offset_y = layer * 20
                
                # Colore layer usando i colori personalizzati, ciclando tra di essi
                # Possiamo associare il colore del layer all'energia di una banda specifica per quel layer
                # Ad esempio, layer 0 -> basse, layer 1 -> medie, layer 2 -> alte, ecc.
                if layer == 0:
                    base_color = theme['colors'][0] # Basse
                elif layer == 1:
                    base_color = theme['colors'][1 % len(theme['colors'])] # Medie
                else: # layers 2+
                    base_color = theme['colors'][2 % len(theme['colors'])] # Alte (or cycle)


                points = []
                for i, amplitude in enumerate(waveform_data):
                    x = int((i / len(waveform_data)) * width)

                    # Effetto 3D: oscillazione verticale + prospettiva, scalato da effective_energy
                    base_y = height // 2 + layer_offset_y
                    wave_y = int(amplitude * height * 0.3 * intensity * np.sin(frame_idx * 0.1 + layer) * effective_energy)
                    y = base_y - wave_y

                    points.append((x, y))

                # Disegna layer
                if len(points) > 1:
                    # Crea poligono riempito per effetto volume
                    fill_points = points + [(width, height), (0, height)]

                    # Colore con trasparenza simulata (PIL non gestisce RGBA per poligoni/linee direttamente)
                    # useremo un colore più scuro per i layer più lontani
                    r, g, b = hex_to_rgb(base_color)
                    dark_factor = 1.0 - (layer / (layers * 1.5)) # Più scuro se più lontano
                    r = int(r * dark_factor)
                    g = int(g * dark_factor)
                    b = int(b * dark_factor)
                    layer_color = f"#{r:02x}{g:02x}{b:02x}"

                    # Disegna riempimento
                    draw.polygon(fill_points, fill=layer_color)

                    # Linea di contorno, scalata da effective_energy (es. spessore linea)
                    line_thickness = max(1, int(2 * effective_energy)) # Più sottile per volume basso
                    for i in range(len(points) - 1):
                        draw.line([points[i], points[i + 1]], fill=base_color, width=line_thickness)

    return img

def create_fluid_dynamics(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Simula dinamica fluidi con la musica."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        # Parametri fluido
        num_waves = 8

        for wave_idx in range(num_waves):
            # Fase della singola onda
            wave_phase = frame_idx * 0.05 * intensity + wave_idx * np.pi / 4

            # Ampiezza basata su energia effettiva
            amplitude = effective_energy * height * 0.2 * intensity

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

            # Colore onda usando i colori personalizzati
            # Possiamo associare i colori alle onde in base alla loro posizione o indice
            if wave_idx % 3 == 0:
                color = theme['colors'][0] # Basse
            elif wave_idx % 3 == 1:
                color = theme['colors'][1 % len(theme['colors'])] # Medie
            else:
                color = theme['colors'][2 % len(theme['colors'])] # Alte


            # Disegna onda come poligono riempito
            if len(points) > 2:
                # Crea forma chiusa per riempimento
                bottom_points = [(width, height), (0, height)]
                wave_polygon = points + bottom_points

                # Trasparenza per sovrapposizione, basata su effective_energy
                r, g, b = hex_to_rgb(color)
                alpha = int(100 + effective_energy * 100)

                # PIL non supporta alpha direttamente, usiamo colori più scuri per trasparenza
                dark_factor = alpha / 255
                r = int(r * dark_factor)
                g = int(g * dark_factor)
                b = int(b * dark_factor)
                wave_color = f"#{r:02x}{g:02x}{b:02x}"

                draw.polygon(wave_polygon, fill=wave_color)

                # Contorno, spessore scalato da effective_energy
                line_thickness = max(1, int(2 * effective_energy))
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i + 1]], fill=color, width=line_thickness)

    return img

def create_geometric_patterns(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                            theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Crea pattern geometrici animati."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)
        center_x, center_y = width // 2, height // 2

        # Pattern concentrici
        num_rings = 6
        base_radius = min(width, height) * 0.05

        for ring in range(num_rings):
            # Raggio anello, pulsazione scalata da effective_energy
            radius = base_radius + ring * (min(width, height) * 0.08)
            radius *= (1 + effective_energy * intensity * 0.5)  # Pulsazione

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

            # Colore usando i colori personalizzati
            # Possiamo associare i colori ai poligoni a seconda del loro 'ring' (frequenza)
            if ring % 3 == 0:
                color = theme['colors'][0] # Basse
            elif ring % 3 == 1:
                color = theme['colors'][1 % len(theme['colors'])] # Medie
            else:
                color = theme['colors'][2 % len(theme['colors'])] # Alte

            # Disegna poligono
            if len(points) > 2:
                # Spessore del contorno o opacità del riempimento scalati da effective_energy
                outline_width = max(1, int(3 * effective_energy)) # Più sottile per volume basso

                if ring % 2 == 0:
                    # Riempito
                    r, g, b = hex_to_rgb(color)
                    alpha = int(100 + effective_energy * 155) # Opacità basata su effective_energy
                    fill_color_rgb = (r, g, b, alpha)
                    draw.polygon(points, fill=fill_color_rgb) # PIL supporta RGBA per fill
                else:
                    # Solo contorno
                    draw.polygon(points, outline=color, width=outline_width)

    return img

def create_neural_network(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                         theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Crea visualizzazione rete neurale pulsante."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)

        # Nodi della rete
        nodes = []
        grid_size = 8

        for i in range(grid_size):
            for j in range(grid_size):
                x = int((i + 1) / (grid_size + 1) * width)
                y = int((j + 1) / (grid_size + 1) * height)

                # Attivazione nodo basata su energia effettiva e posizione
                activation = effective_energy + 0.3 * np.sin(frame_idx * 0.1 + i + j)
                activation = max(0, min(1, activation))

                nodes.append((x, y, activation))

        # Connessioni tra nodi vicini
        for i, (x1, y1, act1) in enumerate(nodes):
            for j, (x2, y2, act2) in enumerate(nodes[i+1:], i+1):
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                # Connetti solo nodi vicini
                if distance < min(width, height) * 0.2:
                    # Intensità connessione
                    connection_strength = (act1 + act2) / 2 * intensity # act1 e act2 già influenzati dall'offset

                    if connection_strength > 0.3:
                        # Spessore linea
                        line_width = max(1, int(connection_strength * 3))

                        # Colore connessione usando i colori personalizzati
                        # Colore basato sull'intensità della connessione o sulla posizione
                        if connection_strength < 0.5: # Connessioni più deboli (basse freq)
                            color = theme['colors'][0]
                        elif connection_strength < 0.75: # Connessioni medie
                            color = theme['colors'][1 % len(theme['colors'])]
                        else: # Connessioni forti (alte freq)
                            color = theme['colors'][2 % len(theme['colors'])]

                        draw.line([x1, y1, x2, y2], fill=color, width=line_width)

        # Disegna nodi
        for x, y, activation in nodes:
            if activation > 0.2:
                node_size = int(3 + activation * 8 * intensity)
                # Colore nodo basato sull'attivazione
                if activation < 0.5:
                    color = theme['colors'][0] # Basse
                elif activation < 0.75:
                    color = theme['colors'][1 % len(theme['colors'])] # Medie
                else:
                    color = theme['colors'][2 % len(theme['colors'])] # Alte

                draw.ellipse([x-node_size, y-node_size, x+node_size, y+node_size], fill=color)

    return img

def create_galaxy_spiral(features: Dict[str, Any], frame_idx: int, resolution: Tuple[int, int],
                        theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Crea spirale galattica in movimento."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1)
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

                # Raggio crescente, scalato da effective_energy
                radius = (i / points_per_arm) * min(width, height) * 0.4
                radius *= (1 + effective_energy * intensity * 0.3)

                # Angolo con rotazione
                angle = t + arm_offset + frame_idx * 0.02 * intensity

                # Posizione
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))

                points.append((x, y))

            # Colore braccio spirale usando i colori personalizzati
            # Basiamo il colore sull'indice del braccio o sull'energia complessiva
            if arm == 0:
                color = theme['colors'][0] # Basse
            elif arm == 1:
                color = theme['colors'][1 % len(theme['colors'])] # Medie
            else:
                color = theme['colors'][2 % len(theme['colors'])] # Alte

            # Disegna spirale come linea continua
            if len(points) > 1:
                for i in range(len(points) - 1):
                    # Intensità diminuisce verso l'esterno
                    intensity_factor = 1 - (i / len(points))
                    line_width = max(1, int(2 + effective_energy * 3 * intensity_factor)) # Spessore scalato dall'offset
                    draw.line([points[i], points[i + 1]], fill=color, width=line_width)

            # Stelle lungo la spirale
            star_density = int(10 + effective_energy * 20) # Densità stelle scalata dall'offset
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
                          theme: Dict[str, Any], intensity: float, fps: int, global_volume_offset: float) -> Image.Image:
    """Crea tempesta elettrica musicale."""
    width, height = resolution
    img = Image.new('RGB', (width, height), theme['background'])
    draw = ImageDraw.Draw(img)

    time_idx = get_time_idx(features, frame_idx, fps)

    if time_idx >= 0:
        current_energy = features['rms_energy'][time_idx]
        effective_energy = np.clip(current_energy * global_volume_offset, 0, 1) # Applica offset
        onset_strength = features['onset_strength'][time_idx]

        # Numero fulmini basato su energia effettiva e onset
        num_lightning = int(1 + onset_strength * 8 + effective_energy * 5)

        for lightning in range(num_lightning):
            # Punto di partenza (alto)
            start_x = random.randint(0, width)
            start_y = random.randint(0, height // 4)

            # Punto di arrivo (basso)
            end_x = start_x + random.randint(-width//4, width//4)
            end_y = random.randint(3*height//4, height)

            # Segmenti fulmine
            current_x, current_y = start_x, start_y

            segments = 8 + int(effective_energy * 12) # Segmenti scalati dall'offset

            for segment in range(segments):
                # Punto successivo
