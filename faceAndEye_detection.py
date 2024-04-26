import numpy as np
import cv2
import os
# from matplotlib import pyplot as plt

# Load the pre-trained Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Face Detection in static Images
def detect_faces_eyes_in_image(image_path,filename):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image from {image_path}")
        return
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the region of interest (ROI) in the grayscale image and color image
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes within the region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Iterate over each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye on the original image
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Add text below the face rectangle with the filename
        text = f"{filename}"
        cv2.putText(img, text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


    # Display the image with detected faces
    cv2.imshow('Detected Faces', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# image_path = 'images/known/messi.jpg'
image_path = 'images/known/Ameer.png'
filename = os.path.splitext(os.path.basename(image_path))[0]
#detect_faces_eyes_in_image(image_path, filename)

#Face Detection in Webcam
def detect_faces_in_webcam():

    # Initialize video capture from the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    # Define the font parameters for the custom text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 0, 0)  # Green color
    font_thickness = 5

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from webcam")
            break  # Break the loop if no frame is captured

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Get the region of interest (ROI) in the grayscale frame and color frame
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes within the region of interest
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Iterate over each detected eye
            for (ex, ey, ew, eh) in eyes:
                # Draw a rectangle around each eye on the original frame
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                # Add text below the face rectangle
                text = "Face"
                cv2.putText(frame, text, (x+10, y+h+40), font, fontScale=font_scale, color=font_color, thickness=font_thickness)

        # Display the frame with detected faces
        cv2.imshow('Face Detection in Webcam', frame)

        # Check for the 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

detect_faces_in_webcam() #calling function

