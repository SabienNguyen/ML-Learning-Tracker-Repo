import cv2
import sys
imagePath = sys.argv[2]
cascPath = sys.argv[3]

faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=4,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
print ("Found {0} face!".format(len(faces)))

for(x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
