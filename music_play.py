from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from win32com.client import Dispatch
from time import sleep

# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]



        mp = Dispatch("WMPlayer.OCX")
        if label=="angry":
          tune = mp.newMedia(r"C:\Users\Ashima Gupta\Downloads\SHAREit\Redmi Note 4\music\1.mp3")
          print("Angry song is playing.")
        elif label=="happy":
          tune = mp.newMedia(r"C:\Users\Ashima Gupta\Downloads\SHAREit\Redmi Note 4\music\2.mp3")
          print("Happy song is playing.")
        elif label=="sad":
          tune = mp.newMedia(r"C:\Users\Ashima Gupta\Downloads\SHAREit\Redmi Note 4\music\3.mp3")
          print("Sad song is playing.")
        elif label=="neutral":
          tune = mp.newMedia(r"C:\Users\Ashima Gupta\Downloads\SHAREit\Redmi Note 4\music\4.mp3")
          print("Nuetral song is playing.")
        elif label=="surprise":
          tune = mp.newMedia(r"C:\Users\Ashima Gupta\Downloads\SHAREit\Redmi Note 4\music\5.mp3")
          print("Surprise song is playing.")

        mp.currentPlaylist.appendItem(tune)
        mp.controls.play()
        sleep(1)
        mp.controls.playItem(tune)
        # to stop playing use
        input("Press Enter to stop playing")
        mp.controls.stop()
    



 
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)
    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()



