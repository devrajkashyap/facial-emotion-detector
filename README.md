# Face Emotion Detection

This repository contains a Python script that uses computer vision techniques and deep learning to detect and recognize facial emotions in real-time using the default camera.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Keras (`keras`)

You can install the required packages using pip:

```
pip install opencv-python numpy keras
```

## Usage

1. Clone the repository or download the script file.

2. Download the pre-trained model `fer2013_mini_XCEPTION.102-0.66.hdf5` place it in the same directory as the script.

3. Run the script using the following command:

   ```
   python face_emotion_detection.py
   ```

4. The script will start capturing video from the default camera and display the live video feed with emotion labels and bounding boxes around detected faces.

5. Press 'q' to exit the program.

## How it works

The script follows the following steps:

1. Loads the pre-trained models:
   - Haar cascade classifier for face detection (`haarcascade_frontalface_default.xml`).
   - Emotion recognition model (`fer2013_mini_XCEPTION.102-0.66.hdf5`).

2. Defines the emotion labels.

3. Starts capturing video from the default camera.

4. In a loop, the script:
   - Reads each video frame.
   - Converts the frame to grayscale for face detection.
   - Detects faces in the frame using the Haar cascade classifier.
   - For each detected face:
     - Extracts the face region of interest (ROI).
     - Preprocesses the face ROI by resizing and normalizing it.
     - Performs emotion recognition on the preprocessed face ROI using the loaded model.
     - Determines the emotion with the highest confidence.
     - Draws the emotion label and bounding box on the frame.
   - If no faces are detected, it displays a "No face detected" label.
   - Displays the resulting frame with emotion labels and bounding boxes.
   - Breaks the loop if 'q' is pressed.

5. Releases the video capture object and closes the windows.


## Acknowledgments

- The emotion recognition model used in this script is based on the [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset.
- The Haar cascade classifier for face detection is provided by OpenCV.
