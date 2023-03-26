# USAGE
# python detect_faces.py --image rooster.jpg 
# import the necessary packages
import numpy as np
import cv2

def crop(image):
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # img_path=str(img_path)
    # image = cv2.imread(img_path)
    image = cv2.resize(image,(600,500))
    #getting height and width of image
    (h, w) = image.shape[:2]

    #Creating a blob from image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
    # loop over the detections
    #print(confidence)
    # initialize variables to keep track of the highest confidence and the bounding box of the face with the highest confidence
    max_confidence = 0
    max_box = None

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > 0.7:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # check if the current face has a higher confidence score than the previous highest
            if confidence > max_confidence:
                max_confidence = confidence
                max_box = box

    # if a face with a high enough confidence score was detected, draw its bounding box and save the image
    if max_box is not None:
        (startX, startY, endX, endY) = max_box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        image=image[startY:endY, startX:endX]
        return image
