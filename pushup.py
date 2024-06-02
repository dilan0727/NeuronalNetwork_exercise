import cv2
import mediapipe as mp
import numpy as np
import math
import pyttsx3
import speech_recognition as sr

# Inicializar la cámara
cap = cv2.VideoCapture(0)
count = 0
direction = 0
form = 0
feedback = "Bad Form. Correct Posture."
exercise = "None"
rest_time = 0

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
def get_position(img, results, draw=True):
    landmark_list = []
    if results.pose_landmarks:
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, c = img.shape
            landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
            landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
            if draw:
                cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255, 0, 0), cv2.FILLED)
    return landmark_list

def get_angle(img, landmark_list, point1, point2, point3, draw=True):
    x1, y1 = landmark_list[point1][1:]
    x2, y2 = landmark_list[point2][1:]
    x3, y3 = landmark_list[point3][1:]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
        cv2.circle(img, (x1, y1), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (75, 0, 130), 2)
        cv2.circle(img, (x2, y2), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (75, 0, 130), 2)
        cv2.circle(img, (x3, y3), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, (75, 0, 130), 2)
        cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return angle

# Reconocimiento de voz
def recognize_voice_command(prompt):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        print(prompt)
        speak(prompt)
        audio = recognizer.listen(source)
    
    try:
        command = recognizer.recognize_google(audio, language='es-ES')
        print(f"Command received: {command}")
        return command
    except sr.UnknownValueError:
        print("Lo siento, no entendí eso.")
        speak("Lo siento, no entendí eso.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

# Función principal para iniciar la cámara y reconocimiento de voz
def main():
    global exercise, rest_time

    # Preguntar por el ejercicio
    exercise = recognize_voice_command("¿Qué ejercicio deseas hacer?")

    # Preguntar por el tiempo de descanso
    while True:
        rest_command = recognize_voice_command("¿Cuánto tiempo de descanso deseas tomar en segundos? Puedes decir un segundo, dos segundos, hasta diez segundos.")
        try:
            rest_time = int(rest_command.split()[0])
            if 1 <= rest_time <= 10:
                break
            else:
                speak("El tiempo de descanso debe ser entre uno y diez segundos.")
        except (ValueError, IndexError):
            speak("Por favor, di un número válido entre uno y diez segundos.")

    # Iniciar la captura de video
    start_camera()

def start_camera():
    global count, direction, form, feedback, exercise, rest_time
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmark_list = get_position(img, results, False)

        if len(landmark_list) != 0:
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
        
        cv2.imshow('Pushup Counter', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

