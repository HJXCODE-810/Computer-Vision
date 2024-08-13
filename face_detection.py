# Face Detection Of Images 
import cv2 as cv


img = cv.imread('J.jpg')
cv.imshow('RDJ',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray RDJ',gray)


haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

print(f'Number of faces found ={len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=1)

cv.imshow('Face Detected',img)

cv.waitKey(0)

