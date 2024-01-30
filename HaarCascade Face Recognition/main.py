import cv2
from PIL import Image

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

# Read the image
image = cv2.imread("test_images/Mulitple Faces.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = cv2.resize(src=gray_image, dsize=(0, 0), fx=0.5, fy=0.5)

# Detect faces in the image
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

print(len(faces))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # print(f"""
    # x : {x}
    # y : {y}
    # w : {w}
    # h : {h}
    # """)
# Display the image with rectangles around the faces
cv2.imshow('Frame', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
