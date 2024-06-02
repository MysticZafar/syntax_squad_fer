import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('FERmodel.h5')

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']


def predict_emotion(face):
    if face.shape[-1] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=-1)
    face = face.reshape(1, 48, 48, 1)
    face = face.astype('float32') / 255.0
    prediction = model.predict(face)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion


def analyze_image(image_path, result_path):
    image = cv2.imread(image_path)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        emotion = predict_emotion(face)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(result_path, image)


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to load video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Emotion Detection - press Esq or Q to cancel', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or Esc key
            break

    cap.release()
    cv2.destroyAllWindows()
