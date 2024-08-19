import cv2
import numpy as np
import face_recognition
import os

# Import the images
path = "C:\\Users\\SANIKA CHAUDHARI\\OneDrive\\Desktop\\imagesss"
images = []  # Create the list
classnames = []
mylist = os.listdir(path)
print(mylist)  # We get the name of the pics in the given folder

for cls in mylist:
    curimg = cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    classnames.append(os.path.splitext(cls)[0])  # Get the name without the extension
print(classnames)

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure that the face was detected
            encodelist.append(encode[0])
    return encodelist

encodelistknown = findencodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

    if encodecurframe:
        for encodeface, faceloc in zip(encodecurframe, facecurframe):
            matches = face_recognition.compare_faces(encodelistknown, encodeface)
            facedis = face_recognition.face_distance(encodelistknown, encodeface)
            #print(facedis)
            matchindex = np.argmin(facedis)

            if matches[matchindex]:
                name = classnames[matchindex].upper()
                print(name)

                # Scale face location back to the original image size
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
