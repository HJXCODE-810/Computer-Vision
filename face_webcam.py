# Face Detection using the WeBCAM
# By Hrithik jerath and Yatin Jindal

import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

video_cap=cv.VideoCapture(0)

def det_bounding_box(vid):
    gray = cv.cvtColor(vid,cv.COLOR_BGR2GRAY)
    face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for (x,y,w,h) in face_rect:
        cv.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        
    return face_rect

while True:
    result,video_frame = video_cap.read()

    if result is False:
        break
    face_det = det_bounding_box(video_frame)

    cv.imshow('My face Detected',video_frame)

    if cv.waitKey(1)& 0xFF == ord('q'):
        break

video_cap.release()
cv.destroyAllWindows()

