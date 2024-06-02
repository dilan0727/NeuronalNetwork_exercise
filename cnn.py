import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Función para cargar los datos de audio
def load_data(data_dir, duration=1, sr=16000):
    labels = ['flexion', 'salto', 'sentadilla']
    X = []
    y = []

    samples_per_track = int(sr * duration)
    n_mfcc = 13

    for label in labels:
        files = os.listdir(os.path.join(data_dir, label))
        for file in files:
            file_path = os.path.join(data_dir, label, file)
            audio, sr = librosa.load(file_path, sr=sr, duration=duration)
            if len(audio) >= samples_per_track:
                audio = audio[:samples_per_track]
            else:
                padding = samples_per_track - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            X.append(mfcc)
            y.append(labels.index(label))

    X = np.array(X)
    y = np.array(y)
    return X, y

# Cargar datos de entrenamiento y prueba
train_dir = 'C:/Users/dilan/Downloads/parr/parcial3/DeepFit-main/DeepFit-main/DataBaseEj/train'
test_dir = 'C:/Users/dilan/Downloads/parr/parcial3/DeepFit-main/DeepFit-main/DataBaseEj/test'

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

# Normalizar los datos
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)

# Añadir dimensiones para los datos de entrada de la CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convertir las etiquetas a una representación categórica
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Imprimir las formas para verificar
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")

# Definir el modelo CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo
model.save('exercise_recognition_model.h5')

# Predicciones en los datos de prueba
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['flexion', 'salto', 'sentadilla'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
