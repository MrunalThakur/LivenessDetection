import cv2
import imutils
from time import time


def cvDnnDetectFaces( opencv_dnn_model, min_confidence=0.5,skip= 30):
    ROI = "ROI/Fake/"
    cap = cv2.VideoCapture(0)
    #time.sleep(2)
    prev = 0
    new = 0
    count = 0
    read = 0
    while(True):
        ret,frame = cap.read()
        # FPS calculator
        # new = time()
        # fps = 1/(new-prev)
        # prev = new
        # fps = int(fps)
        # fps = str(fps)
        if read%skip !=0:
            continue
        if(ret==0):
            print("Error")
            break
        frame = imutils.resize(frame,width=800)
        frame = cv2.flip(frame,1)


        image_height, image_width, _ = frame.shape

        output_image = frame.copy()

        preprocessed_image = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0))

        opencv_dnn_model.setInput(preprocessed_image)



        results = opencv_dnn_model.forward()


        for face in results[0][0]:

            face_confidence = face[2]

            if face_confidence > min_confidence:
                bbox = face[3:]

                x1 = int(bbox[0] * image_width)
                y1 = int(bbox[1] * image_height)
                x2 = int(bbox[2] * image_width)
                y2 = int(bbox[3] * image_height)

                cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0,0,255), thickness=2)

                # Print FPS
                # cv2.putText(output_image, 'fps= '+fps, org= (30, 30),fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=1)

                cv2.putText(output_image, text=str(face_confidence), org=(x1, y1-25),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,
                            color=(255,255,255), thickness=1)
                crop = output_image[y1:y2,x1:x2]
                count = count + 1
                cv2.imwrite(ROI+'Amanface{:d}.jpg'.format(count),crop)






        cv2.imshow('Frame',output_image)
        if cv2.waitKey(24) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()






opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="deploy.prototxt.txt",
                                            caffeModel="res10_300x300_ssd_iter_140000.caffemodel")








# image = cv2.imread('/home/maliciousbrew/Downloads/download.jpeg')
# cv2.imshow('img',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows
skip = 60
min_confidence = 0.99
cvDnnDetectFaces(opencv_dnn_model,min_confidence,skip)
# cv2.imshow('img',output_img)
# cv2.waitKey(0)

# cv2.destroyAllWindows()
