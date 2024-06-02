import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Cargar el modelo entrenado
model_cargado = load_model('modelo_entrenado.h5')

# Definir el diccionario de clases
clases = {0: 'Correr', 1: 'Saltar', 2: 'abdominal', 3: 'flexion', 4: 'sentadilla'}

# Arbol de desición
def Arbol(ruta):
    # Decide entre las clases
    return os.path.splitext(os.path.basename(ruta))[0]

def predecir_ejercicio(video_path, model):
    try:
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))  # Redimensionar a tamaño objetivo
            frames.append(frame)
        cap.release()
        if not frames:
            print("Error: No se pudieron obtener fotogramas del video.")
            return None
        # Rellenar o truncar las secuencias para que todas tengan la misma longitud
        max_len = 50
        frames = frames[:max_len] if len(frames) > max_len else frames + [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(max_len - len(frames))]
        frames = np.array(frames)
        frames = frames.reshape(1, frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3])  # Agregar dimensión de lote
        # Realizar la predicción
        prediction = model.predict(frames)
        exercise_class = np.argmax(prediction)
        class_name = clases[exercise_class]
        return class_name
    except Exception as e:
        print("Error:", e)
        return None

# Utilizar la función para hacer predicciones en un video específico
video_path = 'C:/Users/usuario/OneDrive/Escritorio/I.A/Teoria/3 corte/Action_Recognition-master/data/WIS/video/Datos_pruebas/prueba_correr/correr.avi' 


clase_predicha = predecir_ejercicio(video_path, model_cargado)

# Obtener el nombre de la clase como el último nombre del archivo sin la extensión
clase_predicha = Arbol(video_path)

# Hacer la predicción usando el nombre de la clase obtenido
if clase_predicha is not None:
    print('La clase predicha es:', clase_predicha)



