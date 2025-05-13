from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
import pandas as pd
import pretty_midi
import os

app = FastAPI()

# Define input schema with validation for the range of temperature and num_predictions
class MusicGenerationRequest(BaseModel):
    temperature: float = Field(..., ge=0.1, le=2.0, description="Temperature for music generation (0.1 - 2.0)")
    num_predictions: int = Field(..., ge=10, le=1000, description="Number of predictions (10 - 1000)")

    @validator('temperature')
    def validate_temperature(cls, value):
        if not (0.1 <= value <= 2.0):
            raise ValueError('Temperature must be between 0.1 and 2.0')
        return value

    @validator('num_predictions')
    def validate_num_predictions(cls, value):
        if not (10 <= value <= 1000):
            raise ValueError('Number of predictions must be between 10 and 1000')
        return value

# Fixed path for model weights
LSTM_MODEL_WEIGHTS_PATH = r"checkpoints\lstm_model.weights.h5"
GRU_MODEL_WEIGHTS_PATH = r"checkpoints\gru_model.weights.h5"
CNN_MODEL_WEIGHTS_PATH = r"checkpoints\1dcnn_model.weights.h5"



def load_gru_model():
    if not os.path.exists(GRU_MODEL_WEIGHTS_PATH):
        raise HTTPException(status_code=404, detail="GRU model weights file not found!")
    model = build_gru_model()
    model.load_weights(GRU_MODEL_WEIGHTS_PATH)
    return model


# Build model function
def build_model(input_shape=(100, 3), learning_rate=0.005):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    def mse_with_positive_pressure(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        penalty = tf.reduce_mean(tf.square(tf.minimum(y_pred, 0.)))
        return mse + penalty

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    return model

# Load model weights from fixed path
def load_trained_model():
    if not os.path.exists(LSTM_MODEL_WEIGHTS_PATH):
        raise HTTPException(status_code=404, detail="Weights file not found!")
    model = build_model()
    model.load_weights(LSTM_MODEL_WEIGHTS_PATH)
    return model

# Build CNN model function
def build_cnn_model(input_shape=(100, 3), learning_rate=0.001):
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional blocks
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    def mse_with_positive_pressure(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        penalty = tf.reduce_mean(tf.square(tf.minimum(y_pred, 0.)))
        return mse + penalty

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model

# Predict next note function
def predict_next_note(notes, model, temperature=1.0):
    assert temperature > 0

    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs, verbose=0)

    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    pitch = tf.clip_by_value(pitch, 0, 127)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

# Convert notes to MIDI
def notes_to_midi(note_df, out_file='output.mid', instrument_name='Acoustic Grand Piano'):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    for _, row in note_df.iterrows():
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(row['pitch']),
            start=float(row['start']),
            end=float(row['end'])
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm



def load_cnn_model():
    if not os.path.exists(CNN_MODEL_WEIGHTS_PATH):
        raise HTTPException(status_code=404, detail="CNN model weights file not found!")
    model = build_cnn_model()
    model.load_weights(CNN_MODEL_WEIGHTS_PATH)
    return model

# Build GRU model function
def build_gru_model(input_shape=(100, 3), learning_rate=0.005):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.GRU(128)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    def mse_with_positive_pressure(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        penalty = tf.reduce_mean(tf.square(tf.minimum(y_pred, 0.)))
        return mse + penalty

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    return model




# Music generation endpoint
@app.post("/generate_music/lstm")
async def generate_music(temp: float = 1.0, len: int = 120):
    request = MusicGenerationRequest(temperature=temp, num_predictions=len)
    model = load_trained_model()

    key_order = ['pitch', 'step', 'duration']
    seq_length = 100
    raw_notes = {
        'pitch': np.random.randint(0, 128, size=seq_length),
        'step': np.random.rand(seq_length),
        'duration': np.random.rand(seq_length)
    }

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = sample_notes / np.array([128, 1, 1])

    generated_notes = []
    prev_start = 0
    temperature = request.temperature
    num_predictions = request.num_predictions

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, [input_note], axis=0)
        prev_start = start

    generated_notes_df = pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])

    # Save as MIDI
    out_file = "generated_music.mid"
    notes_to_midi(generated_notes_df, out_file=out_file)
    
    # Serve the file for download
    return FileResponse(out_file, media_type='audio/midi', filename="generated_music.mid")

@app.post("/generate_music/gru")
async def generate_music_gru(temp: float = 1.0, len: int = 120):
    request = MusicGenerationRequest(temperature=temp, num_predictions=len)
    model = load_gru_model()

    key_order = ['pitch', 'step', 'duration']
    seq_length = 100
    raw_notes = {
        'pitch': np.random.randint(0, 128, size=seq_length),
        'step': np.random.rand(seq_length),
        'duration': np.random.rand(seq_length)
    }

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = sample_notes / np.array([128, 1, 1])

    generated_notes = []
    prev_start = 0
    temperature = request.temperature
    num_predictions = request.num_predictions

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, [input_note], axis=0)
        prev_start = start

    generated_notes_df = pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])

    out_file = "generated_music_gru.mid"
    notes_to_midi(generated_notes_df, out_file=out_file)

    return FileResponse(out_file, media_type='audio/midi', filename="generated_music_gru.mid")

@app.post("/generate_music/cnn")
async def generate_music_cnn(temp: float = 1.0, len: int = 120):
    request = MusicGenerationRequest(temperature=temp, num_predictions=len)
    model = load_cnn_model()

    key_order = ['pitch', 'step', 'duration']
    seq_length = 100
    raw_notes = {
        'pitch': np.random.randint(0, 128, size=seq_length),
        'step': np.random.rand(seq_length),
        'duration': np.random.rand(seq_length)
    }

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = sample_notes / np.array([128, 1, 1])

    generated_notes = []
    prev_start = 0
    temperature = request.temperature
    num_predictions = request.num_predictions

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, [input_note], axis=0)
        prev_start = start

    generated_notes_df = pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])

    out_file = "generated_music_cnn.mid"
    notes_to_midi(generated_notes_df, out_file=out_file)

    return FileResponse(out_file, media_type='audio/midi', filename="generated_music_cnn.mid")


