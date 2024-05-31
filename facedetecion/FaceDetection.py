import cv2
import os
import dlib

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = dlib.get_frontal_face_detector()
while (True):
    # face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_id = input('\n Nhap ID khuon mat <return> ==> ')
    if (face_id == '#'):
        break
    else:
        faces_name = input('\n nhap ten khuon mat <return> ==> ')
        print("\n [INFO] Khoi tao camera ...")
        count = 0

        while True:

            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = face_detector.detectMultiScale(gray, 1.3, 5)
            faces = face_detector(gray)
            
            for face in faces:
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()
                cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
                count += 1

                cv2.imwrite("dataset/"+str(faces_name) +'.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y1, x:x1])
                #cv2.imwrite("dataset/Users." + str(face_id) + '.' + str(count) + '.jpg', gray[y:y1, x:x1])
                cv2.imshow('image', img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
            elif count >= 50:
                break
        
        
print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()
