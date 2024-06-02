import os
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf

# Cargar el modelo entrenado
model = load_model('lstm_model.h5')

# Cargar las estadísticas de normalización
mean = np.load('mean.npy')
std = np.load('std.npy')

# Función para extraer características (MFCC)
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Función para preprocesar el audio
def preprocess_audio(audio, sr):
    features = extract_features(audio, sr)
    features = (features - mean) / std
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    return features

# Capturar audio desde el micrófono
def record_audio(duration=1, sr=16000):
    print("Grabando...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Grabación completada")
    audio = np.squeeze(audio)  # Eliminar una dimensión innecesaria
    # Guardar el audio grabado en un archivo para verificación
    sf.write('grabacion.wav', audio, sr)
    return audio, sr

# Predecir el número a partir del audio grabado
def predict_number(audio, sr):
    processed_audio = preprocess_audio(audio, sr)
    prediction = model.predict(processed_audio)
    predicted_label = np.argmax(prediction, axis=1) + 1
    return predicted_label[0]

# Grabar audio y hacer una predicción
audio, sr = record_audio(duration=2, sr=16000)
predicted_number = predict_number(audio, sr)
print(f"El número predicho es: {predicted_number}")
