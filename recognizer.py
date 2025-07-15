import cv2
import numpy as np
from keras.models import load_model

# Load face detector
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Load emotion model
emotion_model = load_model('models/emotion_model.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load age prediction model (Caffe)
age_net = cv2.dnn.readNetFromCaffe(
    'models/age_deploy.prototxt',
    'models/age_net.caffemodel'
)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Start webcam (more stable on Windows with CAP_DSHOW)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Or cv2.CAP_MSMF if DSHOW fails


if not cap.isOpened():
    print("❌ Cannot open webcam. Try restarting your camera or using a different backend.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Get face regions
        face_gray = grayscale[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        # --- Emotion Prediction ---
        try:
            face_emotion = cv2.resize(face_gray, (64, 64))
            face_emotion = face_emotion.reshape(1, 64, 64, 1).astype('float32') / 255.0
            emotion_preds = emotion_model.predict(face_emotion, verbose=0)
            emotion_label = emotion_labels[np.argmax(emotion_preds)]
        except Exception as e:
            emotion_label = "?"
            print("Emotion Error:", e)

        # --- Age Prediction ---
        try:
            face_blob = cv2.dnn.blobFromImage(
                image=face_color,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age_label = AGE_BUCKETS[np.argmax(age_preds[0])]
        except Exception as e:
            age_label = "?"
            print("Age Error:", e)

        # Display
        label = f"{emotion_label}, Age: {age_label}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition - Emotion + Age", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
