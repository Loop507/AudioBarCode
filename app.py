# 🎵 SoundWave Visualizer by Loop507 - Versione Completa Enhanced

import streamlit as st
import numpy as np
import cv2
import librosa
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional
import contextlib
import tempfile

MAX_DURATION = 300  # 5 minuti massimo
MIN_DURATION = 1.0  # 1 secondo minimo
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

FORMAT_RESOLUTIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:3": (800, 600)
}

# Nuove modalità visualizzazione
VISUALIZATION_MODES = {
    "Classic Waveform": "Forma d'onda classica verticale",
    "Dense Matrix": "Matrice densa tipo griglia",
    "Frequency Spectrum": "Spettro a frequenza variabile"
}

# Preset colori per frequenze
FREQUENCY_COLOR_PRESETS = {
    "RGB Classic": {"high": "#FFFF00", "mid": "#00FF00", "low": "#FF0000"},
    "Blue Ocean": {"high": "#00FFFF", "mid": "#0080FF", "low": "#0040FF"},
    "Sunset": {"high": "#FF6600", "mid": "#FF3300", "low": "#CC0000"},
    "Neon": {"high": "#FF00FF", "mid": "#00FFFF", "low": "#FFFF00"},
    "Custom": {"high": "#FFFFFF", "mid": "#808080", "low": "#404040"}
}

def check_ffmpeg() -> bool:
    """Verifica se FFmpeg è disponibile"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def validate_audio_file(uploaded_file) -> bool:
    """Valida il file audio caricato"""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """Carica e processa il file audio con gestione errori migliorata"""
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)  # Standardizza SR
        if len(y) == 0:
            st.error("❌ Il file audio è vuoto o non è stato caricato correttamente.")
            return None, None, None
        
        audio_duration = librosa.get_duration(y=y, sr=sr)
        if audio_duration < MIN_DURATION:
            st.error(f"❌ L'audio deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {audio_duration:.2f}s")
            return None, None, None
        
        if audio_duration > MAX_DURATION:
            st.warning(f"⚠️ Audio troppo lungo ({audio_duration:.1f}s). Verrà troncato a {MAX_DURATION}s.")
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dell'audio: {str(e)}")
        return None, None, None

def estimate_bpm(y, sr) -> float:
    """Stima il BPM dell'audio"""
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        # Converte numpy scalar a float python
        tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
        return tempo_val if tempo_val > 0 else 120.0
    except Exception as e:
        print(f"BPM estimation error: {e}")
        return 120.0

def generate_audio_features(y: np.ndarray, sr: int, fps: int) -> dict:
    """Genera tutte le features audio necessarie con separazione frequenze"""
    try:
        duration = len(y) / sr
        
        # Mel-spectrogram per visualizzazioni generali
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalizza mel-spectrogram
        mel_min, mel_max = float(mel_spec_db.min()), float(mel_spec_db.max())
        if mel_max != mel_min:
            mel_norm = (mel_spec_db - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_spec_db)
        
        # STFT per spectrum analyzer con separazione frequenze
        stft = librosa.stft(y, hop_length=512, n_fft=2048)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Normalizza STFT
        mag_min, mag_max = float(magnitude_db.min()), float(magnitude_db.max())
        if mag_max != mag_min:
            stft_norm = (magnitude_db - mag_min) / (mag_max - mag_min)
        else:
            stft_norm = np.zeros_like(magnitude_db)
        
        # Separazione frequenze in 3 bande
        n_freqs = stft_norm.shape[0]
        freq_low = stft_norm[:n_freqs//3, :]  # Basse: 0-33%
        freq_mid = stft_norm[n_freqs//3:2*n_freqs//3, :]  # Medie: 33-66%
        freq_high = stft_norm[2*n_freqs//3:, :]  # Alte: 66-100%
        
        # RMS energy per linee
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms_min, rms_max = float(rms.min()), float(rms.max())
        rms_norm = (rms - rms_min) / (rms_max - rms_min) if rms_max != rms_min else rms
        
        # Beat tracking migliorato
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        
        # Converti numpy scalars/arrays a tipi Python standard
        tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
        beats_array = beats.astype(np.float32) if hasattr(beats, 'astype') else beats
        
        return {
            'mel_spectrogram': mel_norm,
            'stft_magnitude': stft_norm,
            'freq_low': freq_low,
            'freq_mid': freq_mid,
            'freq_high': freq_high,
            'rms_energy': rms_norm,
            'beats': beats_array,
            'tempo': tempo_val,
            'hop_length': 512,
            'sr': sr,
            'duration': duration
        }
    except Exception as e:
        st.error(f"❌ Errore nell'estrazione features: {e}")
        return None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Converte colore hex in BGR per OpenCV"""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

def cleanup_files(*files):
    """Pulisce i file temporanei"""
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception:
            pass

class VideoGenerator:
    def __init__(self, format_res, level, fps, bg_color, freq_colors, mode):
        self.WIDTH, self.HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.bg_color = bg_color
        self.freq_colors = freq_colors  # Dizionario con colori per high, mid, low
        self.TEMP_VIDEO = "temp_video.mp4"
        self.FINAL_VIDEO = "final_video_with_audio.mp4"
        
        # Parametri basati sul livello
        level_params = {
            "soft": {"density": 40, "sensitivity": 0.6, "thickness": 2},
            "medium": {"density": 60, "sensitivity": 0.8, "thickness": 3},
            "hard": {"density": 80, "sensitivity": 1.0, "thickness": 4}
        }
        self.params = level_params.get(level, level_params["medium"])
        self.MODE = mode
    
    def generate_classic_waveform_frame(self, features: dict, frame_idx: int, beat_intensity: float) -> np.ndarray:
        """Genera frame waveform classica - visualizzazione a barre verticali"""
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        
        # Calcola time index
        time_idx = min(frame_idx, features['stft_magnitude'].shape[1] - 1)
        
        # Parametri per waveform classica
        num_bars = min(self.params['density'] * 2, self.WIDTH // 2)
        margin = self.WIDTH // 20
        available_width = self.WIDTH - 2 * margin
        current_x = margin
        
        for i in range(num_bars):
            if current_x >= self.WIDTH - margin:
                break
            
            # Determina quale banda di frequenza usare
            freq_ratio = i / num_bars
            if freq_ratio < 0.33:  # Basse frequenze
                freq_data = features['freq_low']
                color = self.freq_colors['low']
                base_width = 4  # Linee più spesse per i bassi
            elif freq_ratio < 0.66:  # Medie frequenze
                freq_data = features['freq_mid']
                color = self.freq_colors['mid']
                base_width = 2  # Linee medie
            else:  # Alte frequenze
                freq_data = features['freq_high']
                color = self.freq_colors['high']
                base_width = 1  # Linee sottili per gli alti
            
            # Calcola energia per questa banda
            freq_idx = int((i / num_bars) * freq_data.shape[0])
            freq_idx = min(freq_idx, freq_data.shape[0] - 1)
            
            energy = freq_data[freq_idx, time_idx] if time_idx < freq_data.shape[1] else 0
            energy *= beat_intensity * self.params['sensitivity']
            
            # Larghezza barra basata su energia e frequenza
            bar_width = max(1, int(base_width * (1 + energy * 2)))
            
            # Altezza variabile basata sull'energia
            base_height = int(self.HEIGHT * 0.7)
            bar_height = int(base_height * (0.3 + energy * 0.7))
            
            # Solo se c'è energia sufficiente
            if energy > 0.1:
                y_start = (self.HEIGHT - bar_height) // 2
                y_end = y_start + bar_height
                x_end = min(current_x + bar_width, self.WIDTH - margin)
                
                cv2.rectangle(frame, (current_x, y_start), (x_end, y_end), color, -1)
            
            # Spazio tra barre
            current_x += bar_width + max(1, int(2 * (1 - energy)))
        
        return frame
    
    def generate_dense_matrix_frame(self, features: dict, frame_idx: int, beat_intensity: float) -> np.ndarray:
        """Genera matrice densa - pattern a griglia"""
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        
        time_idx = min(frame_idx, features['stft_magnitude'].shape[1] - 1)
        
        # Griglia di blocchi
        block_size = max(3, int(12 * (1 - self.params['sensitivity'] * 0.3)))
        cols = self.WIDTH // block_size
        rows = self.HEIGHT // block_size
        
        for row in range(rows):
            for col in range(cols):
                # Determina banda di frequenza basata sulla posizione verticale
                freq_ratio = row / rows
                if freq_ratio < 0.33:  # Zona alta = freq acute
                    freq_data = features['freq_high']
                    color = self.freq_colors['high']
                elif freq_ratio < 0.66:  # Zona media = freq medie
                    freq_data = features['freq_mid']
                    color = self.freq_colors['mid']
                else:  # Zona bassa = freq basse
                    freq_data = features['freq_low']
                    color = self.freq_colors['low']
                
                # Calcola indici per i dati audio
                freq_idx = int((col / cols) * freq_data.shape[0])
                freq_idx = min(freq_idx, freq_data.shape[0] - 1)
                
                energy = freq_data[freq_idx, time_idx] if time_idx < freq_data.shape[1] else 0
                energy *= beat_intensity * self.params['sensitivity']
                
                # Disegna blocco se c'è energia sufficiente
                threshold = 0.3 - (beat_intensity - 1.0) * 0.2  # Soglia dinamica con beat
                if energy > threshold:
                    x1 = col * block_size
                    y1 = row * block_size
                    x2 = min(x1 + block_size - 1, self.WIDTH)
                    y2 = min(y1 + block_size - 1, self.HEIGHT)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
        return frame
    
    def generate_frequency_spectrum_frame(self, features: dict, frame_idx: int, beat_intensity: float) -> np.ndarray:
        """Genera spettro di frequenza - barre raggruppate per banda"""
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        
        time_idx = min(frame_idx, features['stft_magnitude'].shape[1] - 1)
        
        # Numero di gruppi di barre
        num_groups = self.params['density'] // 10
        group_width = self.WIDTH // num_groups
        
        for group in range(num_groups):
            group_x = group * group_width
            
            # Ogni gruppo ha 3 sottobarre per le 3 bande di frequenza
            subbar_width = group_width // 4
            
            # Barra per freq basse (più larga, a sinistra)
            low_energy = np.mean(features['freq_low'][:, time_idx]) if time_idx < features['freq_low'].shape[1] else 0
            low_energy *= beat_intensity * self.params['sensitivity']
            if low_energy > 0.1:
                low_height = int(self.HEIGHT * 0.8 * low_energy)
                y_start = self.HEIGHT - low_height
                cv2.rectangle(frame,
                            (group_x, y_start),
                            (group_x + subbar_width * 2, self.HEIGHT),
                            self.freq_colors['low'], -1)
            
            # Barra per freq medie (media, al centro)
            mid_energy = np.mean(features['freq_mid'][:, time_idx]) if time_idx < features['freq_mid'].shape[1] else 0
            mid_energy *= beat_intensity * self.params['sensitivity']
            if mid_energy > 0.1:
                mid_height = int(self.HEIGHT * 0.8 * mid_energy)
                y_start = self.HEIGHT - mid_height
                mid_x = group_x + subbar_width * 2 + 2
                cv2.rectangle(frame,
                            (mid_x, y_start),
                            (mid_x + subbar_width, self.HEIGHT),
                            self.freq_colors['mid'], -1)
            
            # Barra per freq acute (sottile, a destra)
            high_energy = np.mean(features['freq_high'][:, time_idx]) if time_idx < features['freq_high'].shape[1] else 0
            high_energy *= beat_intensity * self.params['sensitivity']
            if high_energy > 0.1:
                high_height = int(self.HEIGHT * 0.8 * high_energy)
                y_start = self.HEIGHT - high_height
                high_x = group_x + subbar_width * 3 + 4
                cv2.rectangle(frame,
                            (high_x, y_start),
                            (high_x + max(1, subbar_width // 2), self.HEIGHT),
                            self.freq_colors['high'], -1)
        
        return frame
    
    def calculate_beat_intensity(self, frame_idx: int, features: dict) -> float:
        """Calcola intensità basata sui beat rilevati"""
        try:
            beats = features.get('beats', [])
            if len(beats) == 0:
                return 1.0
            
            # Converti frame in tempo
            hop_length = features.get('hop_length', 512)
            sr = features.get('sr', 22050)
            current_time = frame_idx * (hop_length / sr)
            
            # Calcola tempi dei beat
            beat_times = np.array(beats) * (hop_length / sr)
            
            # Trova beat più vicino
            if len(beat_times) > 0:
                distances = np.abs(beat_times - current_time)
                closest_beat_dist = float(np.min(distances))
            else:
                closest_beat_dist = 1.0
            
            # Intensità inversamente proporzionale alla distanza dal beat
            beat_window = 60.0 / features.get('tempo', 120) / 4  # Finestra basata su BPM
            if closest_beat_dist < beat_window:
                intensity = 1.0 + 0.8 * (1 - closest_beat_dist / beat_window)
            else:
                intensity = 1.0
            
            return min(1.8, intensity)
        
        except Exception as e:
            print(f"Beat intensity calculation error: {e}")
            return 1.0
    
    def generate_video(self, audio_features: dict, audio_file_path: str) -> bool:
        """Genera video con audio sincronizzato"""
        video_writer = None
        try:
            cleanup_files(self.TEMP_VIDEO, self.FINAL_VIDEO)
            
            # Calcola parametri video
            duration = audio_features['duration']
            total_frames = int(duration * self.FPS)
            
            if total_frames <= 0:
                st.error("❌ Durata video non valida")
                return False
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.TEMP_VIDEO, fourcc, self.FPS,
                                         (self.WIDTH, self.HEIGHT))
            
            if not video_writer.isOpened():
                st.error("❌ Impossibile creare video writer")
                return False
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Genera frames
            for frame_idx in range(total_frames):
                try:
                    # Calcola indice temporale per features audio
                    time_ratio = frame_idx / total_frames
                    
                    # Beat intensity
                    beat_intensity = self.calculate_beat_intensity(frame_idx, audio_features)
                    
                    # Genera frame basato sulla modalità visualizzazione
                    if self.MODE == "Classic Waveform":
                        frame = self.generate_classic_waveform_frame(audio_features, frame_idx, beat_intensity)
                    elif self.MODE == "Dense Matrix":
                        frame = self.generate_dense_matrix_frame(audio_features, frame_idx, beat_intensity)
                    elif self.MODE == "Frequency Spectrum":
                        frame = self.generate_frequency_spectrum_frame(audio_features, frame_idx, beat_intensity)
                    else:
                        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) * 128
                    
                    video_writer.write(frame)
                    
                    # Update progress
                    if frame_idx % 10 == 0:
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Generazione: {frame_idx+1}/{total_frames} frame ({progress*100:.1f}%)")
                
                except Exception as e:
                    st.error(f"❌ Errore frame {frame_idx}: {e}")
                    return False
            
            video_writer.release()
            video_writer = None
            
            # Combina video con audio usando FFmpeg
            status_text.text("🔊 Aggiunta audio al video...")
            
            if not self._add_audio_to_video(self.TEMP_VIDEO, audio_file_path, self.FINAL_VIDEO):
                st.error("❌ Errore nell'aggiunta dell'audio")
                return False
            
            status_text.text("✅ Video completato!")
            return True
        
        except Exception as e:
            st.error(f"❌ Errore generazione video: {e}")
            return False
        finally:
            if video_writer:
                video_writer.release()
    
    def _add_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Combina video e audio usando FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y',  # Sovrascrivi file esistente
                '-i', video_path,  # Video input
                '-i', audio_path,  # Audio input
                '-c:v', 'libx264',  # Codec video
                '-c:a', 'aac',  # Codec audio
                '-strict', 'experimental',
                '-shortest',  # Taglia al più corto
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                st.error(f"FFmpeg error: {result.stderr}")
                return False
            
            return os.path.exists(output_path)
        
        except subprocess.TimeoutExpired:
            st.error("❌ Timeout durante la combinazione audio/video")
            return False
        except Exception as e:
            st.error(f"❌ Errore FFmpeg: {e}")
            return False

def main():
    st.set_page_config(page_title="🎵 SoundWave Visualizer", layout="centered")
    st.title("🎵 SoundWave Visualizer")
    st.markdown("*Trasforma la tua musica in visualizzazioni sincronizzate*")
    
    # Verifica FFmpeg
    if not check_ffmpeg():
        st.error("❌ **FFmpeg non trovato!** Installa FFmpeg per continuare.")
        st.markdown("**Come installare FFmpeg:**")
        st.code("# Ubuntu/Debian:\nsudo apt install ffmpeg\n\n# macOS:\nbrew install ffmpeg\n\n# Windows:\n# Scarica da https://ffmpeg.org/download.html")
        return
    
    uploaded = st.file_uploader("🎵 Carica il tuo file audio", type=["wav", "mp3"])
    
    if uploaded:
        if not validate_audio_file(uploaded):
            return
        
        # Salva file temporaneo
        temp_audio = f"temp_audio_{uploaded.name}"
        try:
            with open(temp_audio, "wb") as f:
                f.write(uploaded.read())
            
            # Processa audio
            with st.spinner("🎵 Caricamento audio..."):
                y, sr, duration = load_and_process_audio(temp_audio)
            
            if y is None:
                return
            
            # Estrai features audio
            with st.spinner("🔍 Analisi audio completa..."):
                features = generate_audio_features(y, sr, fps=30)  # Default FPS per analisi
            
            if features is None:
                return
            
            # Info audio
            tempo_val = float(features['tempo']) if hasattr(features['tempo'], 'item') else features['tempo']
            st.success(f"✅ **Audio elaborato:** {duration:.1f}s | **BPM:** {tempo_val:.1f} | **SR:** {sr}Hz")
            
            # Libera memoria
            del y
            gc.collect()
            
            # Interfaccia controlli
            st.markdown("### ⚙️ Configurazione Video")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fmt_key = st.selectbox("📐 **Formato**", list(FORMAT_RESOLUTIONS.keys()), index=0)
                format_res = FORMAT_RESOLUTIONS[fmt_key]
                st.caption(f"📏 {format_res[0]}×{format_res[1]} px")
            
            with col2:
                level = st.selectbox("⚡ **Intensità**", ["soft", "medium", "hard"], index=1)
                level_desc = {"soft": "Delicato", "medium": "Bilanciato", "hard": "Intenso"}
                st.caption(f"⚡ {level_desc[level]}")
            
            with col3:
                fps = st.selectbox("🎬 **FPS**", [15, 20, 24, 30, 60], index=3)
                st.caption(f"⏱️ {fps} frame/sec")
            
            # Modalità visualizzazione
            st.markdown("### 🎨 Modalità Visualizzazione")
            mode = st.selectbox("✨ **Tipo Visualizzazione**", list(VISUALIZATION_MODES.keys()), index=0)
            st.caption(f"🎨 {VISUALIZATION_MODES[mode]}")
            
            # Controlli colore
            st.markdown("### 🎨 Configurazione Colori")
            
            col4, col5 = st.columns(2)
            with col4:
                bg = st.color_picker("🖤 **Sfondo**", "#000000")
                st.caption("🎨 Colore di fondo")
            
            with col5:
                color_preset = st.selectbox("🌈 **Preset Colori Frequenze**", list(FREQUENCY_COLOR_PRESETS.keys()), index=0)
                st.caption("🎨 Colori per bande di frequenza")
            
            # Colori personalizzati se preset è Custom
            freq_colors = FREQUENCY_COLOR_PRESETS[color_preset].copy()
            if color_preset == "Custom":
                col6, col7, col8 = st.columns(3)
                with col6:
                    freq_colors['high'] = st.color_picker("🔊 **Freq Acute**", "#FFFF00")
                    st.caption("🎵 Linee sottili")
                with col7:
                    freq_colors['mid'] = st.color_picker("🎵 **Freq Medie**", "#00FF00")
                    st.caption("🎵 Linee medie")
                with col8:
                    freq_colors['low'] = st.color_picker("🔉 **Freq Basse**", "#FF0000")
                    st.caption("🎵 Linee spesse")
            else:
                # Mostra preview colori preset
                col6, col7, col8 = st.columns(3)
                with col6:
                    st.color_picker("🔊 Acute", freq_colors['high'], disabled=True)
                with col7:
                    st.color_picker("🎵 Medie", freq_colors['mid'], disabled=True)
                with col8:
                    st.color_picker("🔉 Basse", freq_colors['low'], disabled=True)
            
            # Converti colori hex in BGR
            freq_colors_bgr = {
                'high': hex_to_bgr(freq_colors['high']),
                'mid': hex_to_bgr(freq_colors['mid']),
                'low': hex_to_bgr(freq_colors['low'])
            }
            
            # Generazione video
            st.markdown("---")
            if st.button("🎬 **Genera Video Visualizzazione**", type="primary", use_container_width=True):
                
                # Rigenera features con FPS corretto
                with st.spinner("🔧 Ottimizzazione per FPS selezionato..."):
                    features = generate_audio_features(librosa.load(temp_audio)[0], sr, fps)
                
                with st.spinner("🎬 Generazione video in corso..."):
                    generator = VideoGenerator(
                        format_res, level, fps,
                        hex_to_bgr(bg), freq_colors_bgr,
                        mode
                    )
                    
                    if generator.generate_video(features, temp_audio):
                        st.balloons()
                        st.success("🎉 **Video generato con successo!**")
                        
                        # Download video
                        if os.path.exists(generator.FINAL_VIDEO):
                            with open(generator.FINAL_VIDEO, "rb") as f:
                                video_data = f.read()
                            
                            st.download_button(
                                "⬇️ **Scarica Video**",
                                data=video_data,
                                file_name=f"soundwave_{uploaded.name.split('.')[0]}_{mode.lower().replace(' ', '_')}.mp4",
                                mime="video/mp4",
                                type="primary",
                                use_container_width=True
                            )
                            
                            # Preview video
                            st.video(generator.FINAL_VIDEO)
                        else:
                            st.error("❌ File video non trovato")
                    else:
                        st.error("❌ Errore nella generazione del video")
                
                # Cleanup
                cleanup_files(generator.TEMP_VIDEO, generator.FINAL_VIDEO)
        
        except Exception as e:
            st.error(f"❌ Errore generale: {e}")
        finally:
            cleanup_files(temp_audio)
            gc.collect()
    
    # Info e crediti
    st.markdown("---")
    st.markdown("**🎵 SoundWave Visualizer by Loop507**")
    st.markdown("*Trasforma la tua musica in arte visiva sincronizzata*")
    
    with st.expander("ℹ️ Informazioni"):
        st.markdown("""
        **Modalità Visualizzazione:**
        - **Classic Waveform**: Barre verticali classiche con separazione delle frequenze
        - **Dense Matrix**: Griglia densa con pattern a blocchi
        - **Frequency Spectrum**: Spettro raggruppato per bande di frequenza
        
        **Bande di Frequenza:**
        - 🔊 **Acute (giallo)**: Suoni acuti, linee sottili
        - 🎵 **Medie (verde)**: Voci e strumenti, linee medie  
        - 🔉 **Basse (rosso)**: Bassi e percussioni, linee spesse
        
        **Limiti Tecnici:**
        - Dimensione max: 200MB
        - Durata max: 5 minuti
        - Richiede FFmpeg installato
        """)

if __name__ == "__main__":
    main()
