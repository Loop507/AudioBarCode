# 🎵 SoundWave Visualizer - Artistic Edition

Benvenuto in **SoundWave Visualizer - Artistic Edition**! Questo è un'applicazione Streamlit che trasforma i tuoi file audio in spettacolari visualizzazioni video artistiche, permettendoti di personalizzare lo stile, i colori, lo sfondo e persino aggiungere testo.

## 🌟 Funzionalità

* **Diverse Visualizzazioni Artistiche**: Scegli tra stili unici come "Particle System", "Circular Spectrum", "3D Waveforms", "Fluid Dynamics", "Geometric Patterns", "Neural Network", "Galaxy Spiral", "Lightning Storm" e "Barcode Visualizer".
* **Colori Personalizzabili**: Imposta i colori per lo sfondo e per le basse, medie e alte frequenze.
* **Immagine di Sfondo**: Carica un'immagine personalizzata che verrà utilizzata come sfondo per la visualizzazione.
* **Testo Personalizzato**: Aggiungi un testo a tua scelta, scegliendone la dimensione, il colore e la posizione sullo schermo.
* **Controllo Intensità**: Regola l'intensità del movimento delle visualizzazioni.
* **Offset Volume Globale**: Modifica la reattività complessiva della visualizzazione al volume dell'audio.
* **Formato Video e FPS**: Seleziona il rapporto d'aspetto (es. 16:9, 9:16, 1:1) e il frame rate (FPS) del video finale.
* **Analisi Audio Dettagliata**: Visualizza l'analisi delle bande di frequenza (Bassi, Medi, Alti) del tuo brano.

## 🚀 Come Usare

1.  **Clona il Repository (o scarica i file):**
    Se stai usando Git:
    ```bash
    git clone [https://github.com/IlTuoUsername/soundwave-visualizer.git](https://github.com/IlTuoUsername/soundwave-visualizer.git) # Sostituisci con il link del tuo repository
    cd soundwave-visualizer
    ```
    Altrimenti, scarica semplicemente il file `app.py` e il file `requirements.txt` (che dovrai creare) in una cartella.

2.  **Installa FFmpeg:**
    Questo strumento è essenziale per la creazione dei video.
    * **Windows**: Scarica da [ffmpeg.org](https://ffmpeg.org/download.html) e aggiungilo al tuo PATH.
    * **macOS**: `brew install ffmpeg` (se hai Homebrew)
    * **Linux**: `sudo apt update && sudo apt install ffmpeg` (per Debian/Ubuntu)

3.  **Crea il file `requirements.txt`:**
    Nella stessa cartella di `app.py`, crea un file chiamato `requirements.txt` e incolla il seguente contenuto:
    ```
    streamlit
    numpy>=1.26.0
    librosa>=0.10.0
    matplotlib>=3.7.0
    Pillow>=10.0.0
    scipy>=1.10.0
    ```

4.  **Installa le Dipendenze Python:**
    Apri il tuo terminale o prompt dei comandi, naviga nella directory del progetto e esegui:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Esegui l'Applicazione Streamlit:**
    Sempre dal terminale nella directory del progetto, esegui:
    ```bash
    streamlit run app.py
    ```
    Questo aprirà l'applicazione nel tuo browser web.

6.  **Carica la Musica e Personalizza:**
    * Nell'interfaccia dell'app, carica il tuo file audio (MP3, WAV, FLAC, M4A, AAC).
    * Utilizza le opzioni nella sidebar per selezionare lo stile di visualizzazione, i colori, l'intensità e le opzioni di testo.
    * Puoi anche caricare un'immagine personalizzata per lo sfondo.

7.  **Genera il Video:**
    * Clicca sul pulsante "🚀 Genera Visualizzazione Artistica".
    * L'applicazione elaborerà l'audio, genererà i frame e li assemblerà in un video MP4.
    * Una volta completato, potrai vedere un'anteprima del video e scaricarlo.

## ⚠️ Limiti e Note

* **Durata Massima Audio**: Il video generato avrà una durata massima di 30 minuti. Audio più lunghi verranno tagliati.
* **Dimensione Massima File Audio**: I file audio non devono superare i 200 MB.
* **Prestazioni**: La generazione del video può richiedere del tempo a seconda della durata dell'audio, della risoluzione e della potenza del tuo computer.
* **Font Testo**: L'applicazione tenta di caricare `arial.ttf`. Se non è presente nella stessa directory di `app.py`, verrà utilizzato un font predefinito. Puoi scaricare `arial.ttf` o sostituirlo con un altro font `.ttf` e aggiornare il nome nel codice.

## 📄 Licenza e Attribuzione

Questo software è distribuito gratuitamente sotto i termini della **MIT License** (o puoi scegliere un'altra licenza che preferisci, come la BSD).

**Se utilizzi o ridistribuisci questo software, ti preghiamo di citare l'autore originale: `Loop507`.**





