import cv2 as cv
import face_recognition
from datetime import datetime

# encoding process
def findEncodings(images):
    encodeList = []
    for img in images:
        # convert to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # find the encodings and append them
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# final step: mark the attendance
def markAttendance(name):
    # open attendance csv file
    with open('attendance.csv', 'r+') as f:
        # Move the file pointer to the beginning of the file
        f.seek(0)
        # read all the lines in csv check for any existing attendee
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}\n')


