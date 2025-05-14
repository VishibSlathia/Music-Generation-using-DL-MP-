# Music Generation using Deep Learning

The **Music-Generation-using-DL-MP-** project is a deep learning-based system for generating piano music sequences using MIDI files. It features multiple models including a 1D Convolutional Neural Network (1D CNN), a GRU model, an LSTM model, and a Transformer model with Relative Global Attention, trained on the MAESTRO dataset. The project supports both interactive Jupyter notebook exploration and a FastAPI backend for programmatic access and local MIDI playback using Pygame.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project generates piano music using deep learning models:
- **1D CNN**: Extracts sequential patterns from MIDI events.
- **GRU**: Recurrent model for sequential generation.
- **LSTM**: Long Short-Term Memory network for capturing temporal dynamics.
- **Transformer**: Employs Relative Global Attention for long-range dependencies.

MIDI input is tokenized, modeled, and then converted back to playable MIDI sequences, which can be served via an API and played locally.

## Features

- Generate piano music using 1D CNN, GRU, LSTM, and Transformer models.
- Train models using the MAESTRO dataset.
- Upload and play generated MIDI files using FastAPI + Pygame.
- Visualize generated audio using waveforms and spectrograms.
- Interactive notebooks for training and generation.
- Support for pre-trained weights and saved models.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VishibSlathia/Music-Generation-using-DL-MP-.git
   cd Music-Generation-using-DL-MP-
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install System Dependency (for MIDI playback)**:
   - On Ubuntu:
     ```bash
     sudo apt-get install -y fluidsynth
     ```
   - On Windows/macOS, install a compatible SoundFont and MIDI driver for Pygame playback.

## Usage

### 🚀 FastAPI Mode

1. **Start the API Server**:
   ```bash
   uvicorn generate_music:app --reload
   uvicorn play_mid:app --reload
   ```

2. **Access API Docs**:
   Open your browser and go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. **Upload and Play MIDI**:
   - Use the `/upload-midi/` endpoint to upload `.mid` files.
   - Pygame will play the MIDI locally on your machine.

   Example client:
   ```python
   import requests

   with open("generated_music_gru.mid", "rb") as f:
       response = requests.post("http://127.0.0.1:8000/upload-midi/", files={"file": f})
   print(response.json())
   ```

### 🧠 Notebook Mode

1. **Launch Jupyter Notebooks**:
   ```bash
   jupyter notebook
   ```

2. Open and run:
   - `music_generation.ipynb` for 1D CNN, GRU, and LSTM
   - `transformer.ipynb` for Transformer model

3. To generate and save a MIDI file using Transformer:
   ```python
   output_file = generate_music(model, val_paths[-1], out_dir="tmp/", top_k=15)
   ```

4. To visualize:
   ```python
   y, sr = librosa.load(output_file_wav, sr=None)
   librosa.display.waveshow(y, sr=sr)
   ```

### 💾 Pretrained Models

- Place 1D CNN weights at: `midi_model_checkpoints/1dcnn_model.weights.h5`
- Place Transformer model at: `music_transformer.pkl` or `tmp/music_transformer.keras`

### 📦 Training (Optional)

- For CNN/GRU/LSTM: Use `music_generation.ipynb`
- For Transformer: Use `transformer.ipynb`
- Save your models:
   ```python
   model.save_weights("1dcnn_model.weights.h5")
   with open("music_transformer.pkl", "wb") as f:
       pickle.dump(model, f)
   ```

## Project Structure

```
Music-Generation-using-DL/
├── Demonstration video/                 # Demo video or instructions
│   └── video.txt
├── checkpoints/                         # Model checkpoints (1D CNN, Transformer, LSTM, GRU)
│   └── 1dcnn_model.weights.h5
│   └── lstm_model.weights.h5
│   └── gru_model.weights.h5
│   └── transformer_model.pkl
├── notebooks/                           # Jupyter notebooks for training and generation
│   └── music_generation.ipynb           # Data prep, LSTM, GRU, 1D CNN
│   └── transformer.ipynb                # Transformer
├── generate_music.py                    # Script to generate music via CLI or API
├── play_mid.py                          # Script to play MIDI files using Pygame
├── transformer.py                       # Transformer model implementation
├── requirements.txt                     # All required dependencies
├── README.markdown                      
```

## Dependencies

Install everything via:
```bash
pip install -r requirements.txt
```

Or manually:

- **Python packages**:
  - `tensorflow`, `keras`, `keras_hub`
  - `music21`, `midi_neural_processor`, `pyfluidsynth`
  - `numpy`, `scipy`, `librosa`, `matplotlib`, `pygame`
  - `fastapi`, `uvicorn`, `jupyter`, `pickle-mixin`

- **System dependencies**:
  - `fluidsynth` (Ubuntu) for MIDI rendering
  - SoundFonts/MIDI driver for Pygame playback

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push the branch and open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
