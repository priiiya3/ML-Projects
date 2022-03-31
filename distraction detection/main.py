import dlib
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import yawning

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# load the face detection model
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def eye_aspect_ratio(eye):
    # """ Returns the Eye Aspect Ratio using the coordinates of lankdmarks of the eye. """

	dist_A = distance.euclidean(eye[1], eye[5])
	dist_B = distance.euclidean(eye[2], eye[4])
	dist_C = distance.euclidean(eye[0], eye[3])
	eyeAspectRatio = (dist_A + dist_B) / (2.0 * dist_C)
	return eyeAspectRatio  

flag = 0	
threshold_value = 0.30
frame_check = 20

(left_range, left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(right_range, right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]



# capture the video from camera index: 0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_landmarks, lip_distance = yawning.mouth_open(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceLandmarks = detect(gray, 0)

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
    
    #------- for Drowsiness Detection-----------

    for face in faceLandmarks:
        shape = predict(gray, face)

        #Convert to NumPy array.
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[left_range:left_end]
        rightEye = shape[right_range:right_end]

        # Get the Eye Aspect ratio for left and right eye respectively.
        lefteyeAspectRatio = eye_aspect_ratio(leftEye)
        righteyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (lefteyeAspectRatio + righteyeAspectRatio) / 2.0

        # Set a convex boundary around the eye points. 
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        
        # Draw Contour around the detected eyes
        cv2.drawContours(image, [leftEyeHull], -1, (0, 0, 0), 1)
        cv2.drawContours(image, [rightEyeHull], -1, (0, 0, 0), 1)
        
        # when the eyes start closing.
        if eyeAspectRatio < threshold_value:
            flag += 1

            # if eyes are closed for too long. (that is, if it's more than normal eye blink value)
            if flag >= frame_check:
                cv2.putText(image, "Eyes Closing, STAY FOCUSED !", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # ------for yawning detection------#
        elif lip_distance > 30:
            yawn_status = True   
            cv2.putText(image, "Yawning Detected, STAY FOCUSED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)     
        
        else: # reset the variables
            flag = 0
            yawn_status = False 

    # show the output frame
    cv2.imshow('Final Window', image)

    # Press the esc (escape) key to close the window
    if cv2.waitKey(5) & 0xFF == 27:
        break

# release the camera
cap.release()
cv2.destroyAllWindows()