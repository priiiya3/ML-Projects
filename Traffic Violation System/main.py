"""Object detection using OpenCV and Python."""

import cv2
import numpy as np

# threshold to detect objects. Objects with value lower than 0.5 (50%) will be ignored.
thres = 0.6
nms = 0.2


#  import the classes defined for different objects from 'coco.names' into our List 'classNames'.
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)


# import the configuration file and weights.
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# Set the Detection Model.
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def findObjects(img, draw = True, searchObject = []):

    classIDs, confidences, boundBox = net.detect(img, confThreshold = thres, nmsThreshold=nms)

    #******* On detection of object, we will draw a rectangle around it and diplay it's class and confidence percentange.**********
    
    
    # if name of the objects user wants to detect is not sspecified, then we will detect all the object.
    if len(searchObject) == 0:
        searchObject = classNames
    
    #store the details of detected object in objectInfo list.
    objectInfo, class_id = [], 0
    if len(classIDs) != 0:  # if object is detected : 

        for classID, conf, box in zip(classIDs.flatten(), confidences.flatten(), boundBox):
            
            class_id = classID
            className = classNames[classID - 1]
            if className in searchObject:
                objectInfo.append([box, className])

                if draw:   # if the user wants to detect and display the object info.
                    objectInfo.append([box, className])
                
                    # To draw a rectangle.
                    cv2.rectangle(img, box, color = (0, 0, 0), thickness=1)
                    # to display the class of object detected.
                    cv2.putText(img, className, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 255), 1)
                    # to display the confidence percentage of object detected.
                    # cv2.putText(img, str(round(conf * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    return img, objectInfo, class_id
    


def detect(img):

    """Detects the color of the traffic light in the image passed. Returns the image with a circle drawn on the 
    traffic color detected and color of the traffic light"""

    font = cv2.FONT_HERSHEY_SIMPLEX

    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)

    size = img.shape

    # hough circle detection
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=10, minRadius=0, maxRadius=30)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=5, minRadius=0, maxRadius=30)

    # traffic light detection
    r = 5
    bound = 4.0 / 10
    color_detected = ""
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                color_detected = "Red"
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 80:
                color_detected = "Green"
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                color_detected = "Yellow"
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
    
    return cimg, color_detected



# To capture video from webcam.
cap = cv2.VideoCapture("testing1.mp4")

#  set size of our image window.
cap.set(3, 640)
cap.set(4, 480)

flag, i, count = 0, 0, 0
while True:

    # read frame-by-frame
    success, img = cap.read()

    img = cv2.resize(img, (780, 640), interpolation = cv2.INTER_NEAREST)
    result, objectInfo, class_id = findObjects(img, searchObject=['traffic light'])

    # class id of traffic light in object detetction model is 10.
    if class_id == 10:
        # result = cv2.resize(img, (780, 640), interpolation = cv2.INTER_NEAREST)

        # Extract Region of interest.
        global roi

        # frame[height, width]
        roi = result[:100, 100:] 

        # Traffic Light Detection
        img_roi, colorLabel = detect(roi)
        
        # to give only one warning for each frame even for multiple frames.
        if colorLabel == "Red":
            count += 1
            cv2.putText(result,"Traffic Rule Violation !!!", (200, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
        # cv2.imshow("roi", img_roi)
    
    
    # display the number of traffic lights violated
    cv2.putText(result,"Traffic Violation Counting:{}".format(count), (200, 600), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)


    # Display the resulting frame
    cv2.imshow('result', result)
    
    # Stop when escape key is pressed
    num = cv2.waitKey(10) & 0xff
    if num == 27:
        break

    i += 1

# release the videoCapture object
cap.release()
cv2.destroyAllWindows()
