import os
import numpy as np
import cv2


def load_and_preprocess_images(data_dir):
    emotions = ['angry', 'disgust', 'fear',
                'happy', 'neutral', 'sad', 'surprise']
    width, height = 48, 48
    faces = []
    labels = []

    for idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(idx)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    faces = faces.astype('float32') / 255.0
    labels = np.asarray(labels)
    return faces, labels


if __name__ == "__main__":
    train_faces, train_labels = load_and_preprocess_images('train')
    test_faces, test_labels = load_and_preprocess_images('test')

    np.save('train_faces.npy', train_faces)
    np.save('train_labels.npy', train_labels)
    np.save('test_faces.npy', test_faces)
    np.save('test_labels.npy', test_labels)
