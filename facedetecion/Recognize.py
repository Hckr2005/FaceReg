import cv2
import dlib
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
# cascadePath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascadePath);
faceCascade = dlib.get_frontal_face_detector()

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = [0, 'Phat', 'Phuong', 'Vinh', 'Hieu']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade(gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y1, x:x1])

        if (confidence < 100):
            id = names[id]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y1 - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('nhan dien khuon mat', img)
    k = cv2.waitKey(10) & 0xff  # ESC to escape videoing
    if k == 27:
        break

print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()
