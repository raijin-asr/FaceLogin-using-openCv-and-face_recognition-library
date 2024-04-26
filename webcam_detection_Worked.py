import dlib
import cv2
import face_recognition
import os
import numpy as np
import time
import webbrowser
import subprocess

# Function to load encodings from a folder
def load_encodings_from_folder(folder_path):
    encodings = {}
    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)
        if ext.lower() in ['.jpg', '.png', '.jpeg']:
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encodings[name] = face_encodings[0]  # Assuming only one face per image
    return encodings

# Load known face encodings (you need to prepare this in advance)
known_encodings = load_encodings_from_folder("images/known/")

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()

# Load Camera
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or adjust as needed

start_time = None
login_successful = False
welcome_displayed = False
welcome_start_time = None

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # List to store matching names
    matching_names = []

    if len(faces) == 1:
        # Only proceed if exactly one face is detected
        for face in faces:
            # Get the face coordinates
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

            # Crop the face from the frame
            face_img = frame[y1:y2, x1:x2]

            # Resize face image to a fixed size for face recognition (e.g., 128x128)
            face_img_resized = cv2.resize(face_img, (128, 128))

            # Encode the face using dlib's face recognition model
            face_encodings = face_recognition.face_encodings(face_img_resized)
            
            if face_encodings:
                # Perform face matching with known encodings
                found_name = "Unknown"
                for name, known_encoding in known_encodings.items():
                    # Compare face encodings
                    match = face_recognition.compare_faces([known_encoding], face_encodings[0])
                    if match[0]:
                        found_name = name
                        break

                # Draw rectangle and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, found_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Check if the detected face matches a known face
                if found_name != "Unknown":
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 7:  # show login successful after this time
                        login_successful = True
                else:
                    start_time = None  # Reset start time if unknown face detected

    else:
        # Reset start time if multiple faces detected
        start_time = None
        login_successful = False


    # Display "Login Successful" message after 3 seconds and redirect to website if conditions are met
    if login_successful and not welcome_displayed:
        cv2.putText(frame, "Login Successful!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if time.time() - start_time >= 10: #show welcome message
            welcome_displayed = True
            welcome_start_time = time.time()
    if welcome_displayed:
        # Display "Welcome" message for 5 seconds
        cv2.putText(frame, f"Welcome, {found_name}!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if time.time() - welcome_start_time >= 5:
            
            # Redirect to YouTube website after 5 seconds
            # webbrowser.open("https://www.youtube.com")

            # Open Jupyter Notebook file in web browser
            notebook_file_path = "D:\\PROJECTS\\Python\\Object Detection Project\\ObjectDetection.ipynb"
            subprocess.Popen(['jupyter', 'notebook', notebook_file_path])
           
            # Wait for a few seconds before running the web.py script
            # time.sleep(5)
            # subprocess.Popen(['python', 'webcam_detection_Worked.py'])

            break  # Break the loop after redirecting

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy windows
cap.release()
cv2.destroyAllWindows()