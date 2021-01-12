import cv2

caption = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if(caption.isOpened() == False):
    print("[Error] Problems with video stream")

while(caption.isOpened()):
    ret, frame = caption.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor= 1.3,
            minNeighbors= 5,
	    minSize = (10,10)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

caption.release()
cv2.destroyAllWindows()
