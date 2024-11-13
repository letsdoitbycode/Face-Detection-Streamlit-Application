import streamlit as st
import cv2
import numpy as np

# Load the face detection classifier (Ensure the path to your Haar Cascade is correct)
face_classifier = cv2.CascadeClassifier(r'A:\face-detection\haarcascade_frontalface_default.xml')

# Function to capture and process the webcam feed
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the default webcam
    while True:
        success, frame = cap.read()  # Capture frame-by-frame
        if not success:
            break
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield frame

    cap.release()

# Streamlit UI elements
st.title("Real-Time Face Detection Application")

st.write("Please allow the browser to access your camera to start the video feed.")

st.write("Akshay Bhosale @letsdoibycode")

# Button to start video feed
start_button = st.button("Start Video Feed")

if start_button:
    st.write("Starting the video feed...")

    # Placeholder for the video feed
    stframe = st.empty()

    # Stream the video frames and detect faces
    for frame in generate_frames():
        # Display the frame in real-time with face detection rectangles
        stframe.image(frame, channels="BGR", use_container_width=True)
