#!/usr/bin/python3

from datetime import datetime
import numpy as np
from matplotlib import animation, pyplot as plt


from nicegui import ui

with ui.row():
    with ui.card():
        ui.label('Interactive elements').classes('text-h5')
        with ui.row():
            with ui.column():
                ui.button('Click a picture', on_click=lambda: ui.timer(0.5,Liveness()))
                
                # ui.input(label='Text', value='abc', on_change=lambda e: output.set_text(e.value))
               
        
    with ui.column():
        with ui.card():
            ui.label('Timer').classes('text-h5')
            with ui.row():
                ui.icon('far fa-clock')
                clock = ui.label()
                t = ui.timer(0.1, lambda: clock.set_text(datetime.now().strftime("%X")))
        
    
    with ui.card():
        ui.label('Matplotlib').classes('text-h5')
        with ui.plot(close=False) as plot:
            
            from imutils.video import VideoStream
            from tensorflow.keras.preprocessing.image import img_to_array
            from tensorflow.keras.models import load_model
            import numpy as np
            import imutils
            import pickle
            import time
            import cv2

            #print("[INFO] loading face detector...")
            protoPath = "deploy.prototxt.txt"
            modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
            net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
            # load the liveness detector model and label encoder from disk
            #print("[INFO] loading liveness detector...")
            model = load_model("models/model4")
            le = pickle.loads(open("labelEncoder/le4", "rb").read())
            # initialize the video stream and allow the camera sensor to warmup
            #print("[INFO] starting video stream...")
            fig = plt.figure()
            plt.title('Liveness Detection Window')
            def Liveness():
            
                cap = cv2.VideoCapture(0)
                time.sleep(2.0)

                Confidence = 0.5

                while True:
                    # grab the frame from the threaded video stream and resize it
                    # to have a maximum width of 600 pixels
                    ret,frame = cap.read()
                    frame = imutils.resize(frame, width=600)
                    frame = cv2.flip(frame,1)
                    # grab the frame dimensions and convert it to a blob
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                        (300, 300), (104.0, 177.0, 123.0))
                    # pass the blob through the network and obtain the detections and
                    # predictions
                    net.setInput(blob)
                    detections = net.forward()
                    # loop over the detections
                    for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with the
                        # prediction
                        confidence = detections[0, 0, i, 2]
                        # filter out weak detections
                        if confidence > Confidence:
                            # compute the (x, y)-coordinates of the bounding box for
                            # the face and extract the face ROI
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            # ensure the detected bounding box does fall outside the
                            # dimensions of the frame
                            startX = max(0, startX)
                            startY = max(0, startY)
                            endX = min(w, endX)
                            endY = min(h, endY)
                            # extract the face ROI and then preproces it in the exact
                            # same manner as our training data
                            face = frame[startY:endY, startX:endX]
                            face = cv2.resize(face, (32, 32))
                            face = face.astype("float") / 255.0
                            face = img_to_array(face)
                            face = np.expand_dims(face, axis=0)
                            # pass the face ROI through the trained liveness detector
                            # model to determine if the face is "real" or "fake"
                            preds = model.predict(face)[0]
                            j = np.argmax(preds)
                            label = le.classes_[j]
                            # draw the label and bounding box on the frame
                            label = "{}: {:.4f}".format(label, preds[j])
                            cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY),
                                (0, 0, 255), 2)
                                # show the output frame and wait for a key press
                    with plot:
                        plt.imshow(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                        
                        
                        # cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF
                        # if the `q` key was pressed, break from the loop
                        if key == ord("q"):
                            break
                    cap.release()

   
ui.run()