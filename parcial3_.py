import cv2
import mediapipe as mp
import numpy as np
import math
import pyttsx3
import speech_recognition as sr
import time
from keras.models import load_model

# Inicializar la cámara
cap = cv2.VideoCapture(0)
count = 0
direction = 0
form = 0
feedback = "Bad Form. Correct Posture."
exercise = "None"
rest_time = 0
user_response = ""
target_reps = 10  # Número objetivo de repeticiones antes de descansar
resting = False
series_count = 0  # Contador de series completadas
total_series = 3  # Número total de series

# Cargar modelos
lstm_model = load_model('lstm_model.h5')
exercise_recognition_model = load_model('exercise_recognition_model.h5')
exercise_model = load_model('modelo_entrenado.h5')

# Inicializar el motor de síntesis de voz
engine = pyttsx3.init()

# Función para convertir texto a voz
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Configuración de Mediapipe
def set_pose_parameters():
    mode = False 
    complexity = 1
    smooth_landmarks = True
    enable_segmentation = False
    smooth_segmentation = True
    detectionCon = 0.5
    trackCon = 0.5
    
    mpPose = mp.solutions.pose
    return mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose

mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
pose = mpPose.Pose(mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon)

# Funciones para el procesamiento de pose
def get_pose(img, results, draw=False):  # Cambiado a False para no dibujar
    if results.pose_landmarks:
        if draw:
            mpDraw = mp.solutions.drawing_utils
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img

def get_position(img, results, draw=False):  # Cambiado a False para no dibujar
    landmark_list = []
    if results.pose_landmarks:
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, c = img.shape
            landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
            landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
            if draw:
                # cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255, 0, 0), cv2.FILLED)
                pass
    return landmark_list

def get_angle(img, landmark_list, point1, point2, point3, draw=False):  # Cambiado a False para no dibujar
    x1, y1 = landmark_list[point1][1:]
    x2, y2 = landmark_list[point2][1:]
    x3, y3 = landmark_list[point3][1:]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle

    if draw:
        # cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        # cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
        # cv2.circle(img, (x1, y1), 5, (75, 0, 130), cv2.FILLED)
        # cv2.circle(img, (x1, y1), 15, (75, 0, 130), 2)
        # cv2.circle(img, (x2, y2), 5, (75, 0, 130), cv2.FILLED)
        # cv2.circle(img, (x2, y2), 15, (75, 0, 130), 2)
        # cv2.circle(img, (x3, y3), 5, (75, 0, 130), cv2.FILLED)
        # cv2.circle(img, (x3, y3), 15, (75, 0, 130), 2)
        # cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        pass
    return angle

# Reconocimiento de voz
def recognize_voice_command(prompt, expected_phrases):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        print(prompt)
        speak(prompt)
        audio = recognizer.listen(source)
    
    try:
        command = recognizer.recognize_google(audio, language='es-ES')
        print(f"Command received: {command}")
        
        for phrase in expected_phrases:
            if phrase in command.lower():
                return phrase
        
        return None
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        speak("Lo siento, no entendí eso.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

# Función principal para iniciar la cámara y reconocimiento de voz
def main():
    global exercise, rest_time, user_response

    # Preguntar por el ejercicio
    # Simulando el uso del modelo de reconocimiento de ejercicio por voz
    exercise = recognize_voice_command("¿Qué ejercicio deseas hacer? ", ["flexion", "salto", "sentadilla"])
    if not exercise:
        exercise = "None"

    # Preguntar por el tiempo de descanso
    expected_phrases = [f"{i} segundo" if i == 1 else f"{i} segundos" for i in range(1, 11)]
    
    while True:
        rest_command = recognize_voice_command("¿Cuánto tiempo de descanso deseas tomar? Di un segundo, dos segundos, hasta diez segundos.", expected_phrases)
        if rest_command:
            try:
                rest_time = int(rest_command.split()[0])
                if 1 <= rest_time <= 10:
                    break
                else:
                    speak("El tiempo de descanso debe ser entre un segundo y diez segundos.")
            except ValueError:
                speak("Por favor, di un número válido.")
        else:
            speak("No entendí el tiempo de descanso. Por favor, inténtalo de nuevo.")

    # Iniciar la captura de video
    start_camera()

def start_camera():
    global count, direction, form, feedback, exercise, rest_time, user_response, resting, series_count, total_series
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = get_pose(img, results, False)  # Cambiado a False para no dibujar
        landmark_list = get_position(img, results, False)  # Cambiado a False para no dibujar

        if len(landmark_list) != 0 and not resting:
           
            elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
            shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
            hip_angle = get_angle(img, landmark_list, 11, 23, 25)

            pushup_success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
            pushup_progress_bar = np.interp(elbow_angle, (90, 160), (380, 50))

            if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
                form = 1

            if form == 1:
                if pushup_success_percentage == 0:
                    if elbow_angle <= 90 and hip_angle > 160:
                        feedback = "Go Up"
                        if direction == 0:
                            count += 0.5
                            direction = 1
                    else:
                        feedback = "Bad Form. Correct Posture."
                
                if pushup_success_percentage == 100:
                    if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
                        feedback = "Go Down"
                        if direction == 1:
                            count += 0.5
                            direction = 0
                    else:
                        feedback = "Bad Form. Correct Posture."
            
            print(count)

            if form == 1:
                cv2.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
                cv2.rectangle(img, (580, int(pushup_progress_bar)), (600, 380), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(pushup_success_percentage)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.rectangle(img, (0, 380), (100, 480), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
            cv2.rectangle(img, (500, 0), (640, 40), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, feedback, (500, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Mostrar el comando de voz en la ventana
        cv2.putText(img, f'Ejercicio: {exercise}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f'Tiempo de descanso: {rest_time} seg', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f'Serie: {series_count}/{total_series}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Pushup Counter', img)
        
        if count >= target_reps and not resting:
            speak(f"Descanso de {rest_time} segundos")
            resting = True
            time.sleep(rest_time)
            series_count += 1
            if series_count >= total_series:
                speak("Ejercicio completado.")
                break
            count = 0
            resting = False

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
