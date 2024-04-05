import numpy as np
import cv2 as cv
import face_recognition
import os
from datetime import datetime
from utils import findEncodings, markAttendance

# initialize the webcam
cap = cv.VideoCapture(0)
FRAME_WIDTH = 480  
FRAME_HEIGHT = 640
cap.set(10, 0)

path = './images_attendance'
images = []
classNames = []
myList = os.listdir(path)

# Define the codec and create VideoWriter object
out = cv.VideoWriter('face_recognizer.avi',
                     cv.VideoWriter_fourcc(*'XVID'), 
                     20.0, 
                     (FRAME_HEIGHT, FRAME_WIDTH)) # final frame size

# get the name of the attendance
for cls in myList:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

# encode the faces 
encodeListKnown = findEncodings(images)
print('Encoding is done!')

# find the matches between the encoding
while True:
    success, frame = cap.read()
    # reduce the size of img for faster process
    frame_reduced = cv.resize(frame, (0,0), None, 0.25,0.25)
    # convert to RGB
    frame_reduced = cv.cvtColor(frame_reduced, cv.COLOR_BGR2RGB)
    # find the location of the face
    facesCurrentFrame = face_recognition.face_locations(frame_reduced)
    # find encoding of the webcam
    encodeCurrentFrame = face_recognition.face_encodings(frame_reduced, facesCurrentFrame)

    # find the matches (iterate through all the faces in the current frame)
    for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
    # compare with the encodeFace
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) 
        # compare the faces and return a list of distance numbers
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # find the lowest distance index to be the match
        matchIndex = np.argmin(faceDistance)

        # display bounding box around them
        if matches[matchIndex]:
            # get the name
            name = classNames[matchIndex].upper()
            #print(name)
            # get the face location
            y1,x2,y2,x1 = faceLoc
            # since initially the image was scaled down, convert it back here
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            # draw rectangle box around the face
            cv.rectangle(frame, 
                         (x1, y1), 
                         (x2, y2),
                         (255,0,255),
                         2)
            # draw rectangle box for the text
            cv.rectangle(frame, 
                         (x1, y2-20), 
                         (x2, y2),
                         (255,0,255),
                         cv.FILLED)   
            # put the text name
            cv.putText(frame,
                       f'{name}',
                       (x1+6,y2-6),
                       cv.FONT_HERSHEY_COMPLEX,
                       0.5,
                       (0,0,0),
                       2)  
            # send the name to the mark attendance
            markAttendance(name)    

    # Write the frame to the output video
    out.write(frame)
    cv.imshow('webcam', frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()





