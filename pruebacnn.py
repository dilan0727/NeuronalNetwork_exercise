import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd

# Función para capturar audio en tiempo real
def record_audio(duration=1, sr=16000):
    print("Habla ahora...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Esperar hasta que la grabación esté completa
    return audio.T

# Función para preprocesar el audio capturado
def preprocess_audio(audio, duration=1, sr=16000):
    samples_per_track = int(sr * duration)
    if audio.size < samples_per_track:
        padding = samples_per_track - audio.size
        audio = np.pad(audio, (0, padding), mode='constant')
    else:
        audio = audio[:samples_per_track]

    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    # Agregar dimensiones para los canales y el lote
    mfcc = np.expand_dims(mfcc, axis=0)  # forma (1, n_mfcc)
    mfcc = np.expand_dims(mfcc, axis=-1)  # forma (1, n_mfcc, 1)
    return mfcc

# Cargar el modelo entrenado
model = tf.keras.models.load_model('exercise_recognition_model.h5')

# Clases de ejercicio
exercise_classes = ['flexion', 'salto', 'sentadilla']

# Probar el modelo con audio en tiempo real
while True:
    input("Presiona Enter para comenzar a grabar, o Ctrl+C para salir...")
    audio = record_audio()
    processed_audio = preprocess_audio(audio)
    
    # Realizar la predicción
    prediction = model.predict(processed_audio)
    predicted_class = exercise_classes[np.argmax(prediction)]
    
    print(f"Predicción: {predicted_class}")

