import cv2
import face_recognition

img = cv2.imread("images/known/messi.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img = cv2.imread("images/unknown/messi1.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)