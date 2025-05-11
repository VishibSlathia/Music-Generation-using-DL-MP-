# Music Generation using Deep Learning

The **Music-Generation-using-DL-MP-** project is a deep learning-based application for generating piano music sequences using MIDI files. It implements two models: a 1D Convolutional Neural Network (1D CNN) and a Transformer model with Relative Global Attention, leveraging the MAESTRO dataset. The project uses the `music21` and `midi_neural_processor` libraries for MIDI processing and TensorFlow/Keras for model development. The provided Jupyter notebooks (`music_generation.ipynb` and `transformer.ipynb`) enable data preprocessing, model training, and music generation.

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
This project generates piano music using two deep learning models:
- **1D CNN**: Processes MIDI sequences for music generation (implemented in `music_generation.ipynb`).
- **Transformer**: Utilizes Relative Global Attention for sequence modeling, trained on the MAESTRO dataset (implemented in `transformer.ipynb`).
The models process MIDI files to predict and generate new musical sequences, with support for MIDI playback and visualization.

## Features
- Generates piano music using 1D CNN and Transformer models.
- Processes MIDI files with `music21` and `midi_neural_processor`.
- Downloads and preprocesses the MAESTRO dataset for training.
- Supports loading pre-trained models and generating MIDI sequences.
- Visualizes generated audio with waveforms and spectrograms.
- Saves models using pickle for reuse.

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VishibSlathia/Music-Generation-using-DL-MP-.git
   cd Music-Generation-using-DL-MP-
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install required Python packages:
   ```bash
   pip install tensorflow music21 midi_neural_processor keras_hub keras>=3.6.0 pyfluidsynth scipy numpy jupyter librosa matplotlib
   ```
   Install FluidSynth for MIDI playback (Ubuntu-based systems):
   ```bash
   sudo apt-get install -y fluidsynth
   ```
   For other operating systems, refer to [FluidSynth documentation](https://github.com/FluidSynth/fluidsynth).

4. **Download MAESTRO Dataset**:
   The `transformer.ipynb` notebook automatically downloads the MAESTRO dataset to `datasets/maestro/` using:
   ```python
   paths = sorted(download_maestro(output_dir="datasets/maestro"))
   ```
   Ensure sufficient disk space (~600 MB).

5. **Obtain Pre-trained Model Weights**:
   For the 1D CNN, place `1dcnn_model.weights.h5` in `midi_model_checkpoints/` (as referenced in `music_generation.ipynb`). For the Transformer, the model is saved as `music_transformer.pkl` or `tmp/music_transformer.keras`. If unavailable, train the models as described in [Usage](#usage).

## Usage
1. **Prepare the Environment**:
   Ensure dependencies are installed and the MAESTRO dataset is downloaded.

2. **Run the Jupyter Notebooks**:
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

   - **For 1D CNN Model (`music_generation.ipynb`)**:
     - Open `music_generation.ipynb`.
     - Execute cells to install FluidSynth, load the pre-trained model, and generate music:
       ```python
       model = tf.keras.Model(inputs, outputs)
       model.load_weights('/content/drive/MyDrive/midi_model_checkpoints/1dcnn_model.weights.h5')
       ```
     - Generate MIDI files using the provided generation cells.

   - **For Transformer Model (`transformer.ipynb`)**:
     - Open `transformer.ipynb`.
     - Execute cells to:
       - Install dependencies and FluidSynth.
       - Download and preprocess the MAESTRO dataset.
       - Create and train the Transformer model with `MidiDataset` and `RelativeGlobalAttention`.
       - Generate music using a seed MIDI file:
         ```python
         output_file = generate_music(model, val_paths[-1], out_dir="tmp/", top_k=15)
         ```
       - Visualize the output with waveforms and spectrograms using `librosa` and `matplotlib`.
       - Save the model:
         ```python
         with open("music_transformer.pkl", "wb") as file:
             pickle.dump(model, file)
         ```

3. **Train Models** (if needed):
   - For the 1D CNN, update the data path in `music_generation.ipynb` and run training cells. Save weights:
     ```python
     model.save_weights('1dcnn_model.weights.h5')
     ```
   - For the Transformer, run the training cells in `transformer.ipynb` to train on the MAESTRO dataset. Save the model as shown above.

4. **Generate and Visualize Music**:
   - Generate MIDI files using either notebook.
   - Convert MIDI to WAV and visualize using `transformer.ipynb`:
     ```python
     y, sr = librosa.load(audio_path, sr=None)
     librosa.display.waveshow(y, sr=sr)
     ```

## Project Structure
```
Music-Generation-using-DL-MP-/
├── data/                           # Directory for MIDI files (optional)
├── datasets/maestro/               # MAESTRO dataset directory
├── midi_model_checkpoints/         # Pre-trained 1D CNN model weights
├── tmp/                            # Generated MIDI and WAV files
├── music_generation.ipynb          # 1D CNN model notebook
├── transformer.ipynb               # Transformer model notebook
├── music_transformer.pkl           # Saved Transformer model
├── README.md                       # Project documentation
```

## Dependencies
- **Python Packages**:
  - `tensorflow`, `keras>=3.6.0`, `keras_hub`: For model development.
  - `music21`, `midi_neural_processor`, `pyfluidsynth`: For MIDI processing.
  - `numpy`, `scipy`, `librosa`, `matplotlib`: For data processing and visualization.
  - `jupyter`: For running notebooks.
  Install using:
  ```bash
  pip install tensorflow music21 midi_neural_processor keras_hub keras>=3.6.0 pyfluidsynth scipy numpy jupyter librosa matplotlib
  ```
- **System Dependencies**:
  - `fluidsynth`: For MIDI playback (install via `sudo apt-get install -y fluidsynth` on Ubuntu).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Ensure code follows the project's style and includes documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.