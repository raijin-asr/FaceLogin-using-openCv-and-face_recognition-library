import dlib
import cv2
import face_recognition
import os
import numpy as np

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

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

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

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy windows
cap.release()
cv2.destroyAllWindows()