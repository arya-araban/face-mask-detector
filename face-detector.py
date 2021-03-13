import cv2
import sys
import keyboard

# considering what to print
whatToPrint = 'No Mask'
font = cv2.FONT_ITALIC
color = (0, 0, 255)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show text for the status of wearing - not wearing mask
    cv2.putText(frame, whatToPrint, (10, 450), font, 1, color, 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Mask Detector', frame)
    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord('q'):
        break
    if pressedKey == ord("p"):
        cv2.imshow("cropped", crop_img)
        whatToPrint = "mask"
        color = (0, 255, 0)
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
