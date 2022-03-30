""" Module to detect human faces using openCV and mediapipe. """

import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCount = 0.5):
        
        # using the facedetection module of the mediapipe
        self.minDetectionCount = minDetectionCount
        self.mpFaceDetection = mp.solutions.face_detection
        # to draw shapes on our frames
        self.mpDraw = mp.solutions.drawing_utils    
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCount)


    def facesFound(self, image, draw = True):

        #  convert the image from BGR to RGB form.
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        boundBoxArray = []

        if self.results.detections:
            
            for id, detection in enumerate(self.results.detections):
               
                boundBox_class = detection.location_data.relative_bounding_box 
                img_ht, img_wt, img_chnl = image.shape

                #  Dimensions of the bounding box.
                boundBox = int(boundBox_class.xmin * img_wt), int(boundBox_class.ymin * img_ht), \
                        int(boundBox_class.width * img_wt), int(boundBox_class.height * img_ht) 
                
                boundBoxArray.append([id, boundBox, detection.score])

                image = self.edgeDesigns(image, boundBox)

                # here "detection.score" tells how sure the detection is.
                cv2.putText(image, f'{int(detection.score[0] * 100)}%',
                            (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (50, 255, 0), 2)

        return (image, boundBoxArray)


    def edgeDesigns(self, image, boxes, l = 30, thickness = 8, rec_thick=1):
        # accessing the edges of the boxes.
        x, y, w, h = boxes

        x1, y1 = x + w, y + h

        # This will draw rectangular boxes around the faces detected.
        cv2.rectangle(image, boxes, (8, 255, 5), rec_thick)
        
        # Adding extra thickness at the edges of the bounding boxes.

        # Top Left  x,y
        cv2.line(image, (x, y), (x + l, y), (255, 120, 25), thickness)
        cv2.line(image, (x, y), (x, y+l), (255, 120, 25), thickness)
        # Top Right  x1,y
        cv2.line(image, (x1, y), (x1 - l, y), (255, 120, 25), thickness)
        cv2.line(image, (x1, y), (x1, y+l), (255, 120, 25), thickness)
        # Bottom Left  x,y1
        cv2.line(image, (x, y1), (x + l, y1), (255, 120, 25), thickness)
        cv2.line(image, (x, y1), (x, y1 - l), (255, 120, 25), thickness)
        # Bottom Right  x1,y1
        cv2.line(image, (x1, y1), (x1 - l, y1), (255, 120, 25), thickness)
        cv2.line(image, (x1, y1), (x1, y1 - l), (255, 120, 25), thickness)


        return image


def main():
    
    # To capture video from webcam.
    Vcaptures = cv2.VideoCapture(0)
    detector = FaceDetector()

    prevTime = 0
    while True:
        status, frame = Vcaptures.read()

        # The facesFound module will return the detected faces. 
        frame, boxes = detector.facesFound(frame)

        curTime = time.time()
        fps = 1/(curTime - prevTime)
        prevTime = curTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (5, 8, 7), 2)
        
        
        
        # Display the resulting frame
        cv2.imshow('frames', frame)
        
        # Stop when escape key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # release the videoCapture object.
    Vcaptures.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
