# NeuronalNetwork_exercise
Push-Up Exercise Tracker with Pose Estimation and Voice Recognition

This project leverages computer vision, pose estimation, and voice recognition to create an intelligent push-up exercise tracker. Using OpenCV, MediaPipe, and pre-trained machine learning models, this application provides real-time feedback on exercise form, counts repetitions, and integrates rest intervals with user-defined durations.

Features:

Real-Time Pose Estimation: Utilizes MediaPipe for accurate body pose detection to analyze exercise form.
Voice Interaction: Employs the speech_recognition library to recognize voice commands for selecting exercises and setting rest periods.
Text-to-Speech Feedback: Provides real-time auditory feedback on exercise form and progress using the pyttsx3 library.
Exercise Tracking: Counts repetitions and tracks multiple sets, providing visual and audio feedback on progress and form.
Machine Learning Models: Incorporates pre-trained LSTM and CNN models to recognize exercises and provide tailored feedback.
Dependencies:

OpenCV
MediaPipe
NumPy
Pyttsx3
SpeechRecognition
Keras
Usage:

Initialize Camera: Start the application to initialize the camera for pose estimation.
Voice Commands: Use voice commands to select the exercise type and set the rest time between sets.
Exercise Tracking: Perform the exercise while receiving real-time feedback and progress tracking.
Rest Intervals: The application will automatically count rest intervals and notify you when to resume.
This project aims to enhance the workout experience by providing intelligent, real-time assistance, ensuring proper form, and structured exercise routines.
