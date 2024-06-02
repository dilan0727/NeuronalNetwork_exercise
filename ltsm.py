import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Rutas a las bases de datos de audio
train_database_path = 'C:/Users/dilan/Downloads/parr/parcial3/DeepFit-main/DeepFit-main/DataBaserep/train'
test_database_path = 'C:/Users/dilan/Downloads/parr/parcial3/DeepFit-main/DeepFit-main/DataBaserep/test'

# Función para cargar los datos de audio y sus etiquetas
def load_data(database_path):
    labels = []
    audios = []
    
    valid_folders = [str(i) for i in range(1, 11)]
    
    for folder in os.listdir(database_path):
        folder_path = os.path.join(database_path, folder)
        if os.path.isdir(folder_path) and folder in valid_folders:
            label = int(folder)
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.endswith('.wav'):
                    audio, sr = librosa.load(file_path, sr=None)
                    audios.append(audio)
                    labels.append(label)
    
    return np.array(audios), np.array(labels)

# Cargar los datos de entrenamiento y sus etiquetas
train_audios, train_labels = load_data(train_database_path)

# Cargar los datos de prueba y sus etiquetas
test_audios, test_labels = load_data(test_database_path)

# Dividir datos de entrenamiento en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(train_audios, train_labels, test_size=0.2, random_state=42)

# Extracción de características (MFCC)
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Aplicar extracción de características a los datos
X_train = np.array([extract_features(x, sr) for x, sr in zip(X_train, [16000]*len(X_train))])
X_val = np.array([extract_features(x, sr) for x, sr in zip(X_val, [16000]*len(X_val))])
X_test = np.array([extract_features(x, sr) for x, sr in zip(test_audios, [16000]*len(test_audios))])

# Normalización de los datos
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Ajustar las dimensiones para LSTM
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Convertir etiquetas a one-hot encoding
y_train = to_categorical(y_train - 1, num_classes=10)
y_val = to_categorical(y_val - 1, num_classes=10)
y_test = to_categorical(test_labels - 1, num_classes=10)

# Definir modelo LSTM
model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluar modelo en los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Guardar modelo entrenado
model.save('lstm_model.h5')

# Guardar las estadísticas de normalización
np.save('mean.npy', mean)
np.save('std.npy', std)

# Predicciones en los datos de prueba
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[str(i) for i in range(1, 11)])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
