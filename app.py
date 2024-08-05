import cv2
import streamlit as st
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf

# Add Bootstrap CSS
st.markdown('<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">', unsafe_allow_html=True)

# Initialize MediaPipe face landmark model
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# Initialize Streamlit app
st.title("Webcam Facial Landmarks")
start = st.button('Start Camera')
stop = st.button('Stop Camera')  # Add the "Stop Camera" button

# Initialize heart rate history DataFrame
heart_rate_df = pd.DataFrame(columns=["Timestamp", "BPM"])

# Initialize heart rate chart outside the loop
plt.figure(figsize=(10, 6))
plt.xlabel("Time")
plt.ylabel("Heart Rate (BPM)")
plt.title("Heart Rate History")
plt.grid(True)
heart_rate_line, = plt.plot([], [], marker="o", linestyle="--", color="purple")

if start:
    streamlit_image = st.image([])
    camera = cv2.VideoCapture(0)

    while True:
        check, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect facial landmarks
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract specific facial landmarks (e.g., points 27-30 for forehead)
                forehead_landmarks = [
                    face_landmarks.landmark[27],  # Point 27
                    face_landmarks.landmark[28],  # Point 28
                    face_landmarks.landmark[29],  # Point 29
                    face_landmarks.landmark[30]   # Point 30
                ]

                # Calculate forehead intensity
                forehead_intensity = [frame[int(landmark.y * frame.shape[0]), int(landmark.x * frame.shape[1]), 0]
                                      for landmark in forehead_landmarks]

                # Update heart rate history DataFrame
                if len(forehead_intensity) >= 3:
                    last_bpm = forehead_intensity[-1]
                    heart_rate_df.loc[len(heart_rate_df)] = [pd.Timestamp.now(), last_bpm]

                # Determine stress level based on intensity
                if max(forehead_intensity) < 50:
                    stress_level = "Low Stress"
                elif max(forehead_intensity) < 100:
                    stress_level = "Moderate Stress"
                else:
                    stress_level = "High Stress"

                # Overlay stress level on the frame
                cv2.putText(frame, stress_level, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            streamlit_image.image(frame, channels="RGB", use_column_width=True)

            # Update heart rate chart with new data
            heart_rate_line.set_data(heart_rate_df["Timestamp"], heart_rate_df["BPM"])
            plt.xlim(heart_rate_df["Timestamp"].min(), heart_rate_df["Timestamp"].max())
            plt.ylim(heart_rate_df["BPM"].min() - 10, heart_rate_df["BPM"].max() + 10)
            st.pyplot(plt)  # Display the updated chart

            if stop:  # Check if the "Stop Camera" button is pressed
                break  # Exit the loop

    # Release the camera when the app stops
    camera.release()

# Clean up after the loop
st.write("Camera stopped.")
