import cv2
import tensorflow as tf
import numpy as np
from keras_preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def preprocess_image(img):
    im = cv2.resize(img, (224, 224))
    im = img_to_array(im)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    return im


FRAME_SKIPS = 20
cur_frame = 0

loaded_model = tf.keras.models.load_model('mask_recog.h5')

# considering what to print
whatToPrint = 'No Mask'
color = (0, 0, 255)
font = cv2.FONT_ITALIC

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
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 50), (0, 255, 0), 2)
    # show text for the status of wearing - not wearing mask
    cv2.putText(frame, whatToPrint, (10, 450), font, 1, color, 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Mask Detector', frame)
    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord('q'):
        break

    if cur_frame % FRAME_SKIPS == 0:
        cur_frame = 0

        try:
            crop_img = preprocess_image(crop_img)
            prediction = loaded_model.predict(crop_img)
        except:
            pass

        if (np.argmax(prediction) == 1):
            whatToPrint = 'No Mask'
            color = (0, 0, 255)
        else:
            whatToPrint = "mask"
            color = (0, 255, 0)

    if pressedKey == ord("p"):
        crop_img = preprocess_image(crop_img)
        prediction = loaded_model.predict(crop_img)
        print(prediction)

    cur_frame += 1
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
