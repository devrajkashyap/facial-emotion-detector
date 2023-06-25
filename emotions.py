import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')

# Define the emotions labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read the video frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi_gray = gray[y:y + h, x:x + w]
            face_roi_gray = cv2.resize(face_roi_gray, (64, 64))
            face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)

            # Perform emotion recognition on the face ROI
            face_blob = np.expand_dims(face_roi_gray, axis=0) / 255.0
            emotion_predictions = emotion_model.predict(face_blob)

            # Find the emotion with maximum confidence
            emotion_index = np.argmax(emotion_predictions[0])
            emotion_label = emotions[emotion_index]

            # Draw the emotion label on the frame
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # No face detected, display label
        cv2.putText(frame, 'No face detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
