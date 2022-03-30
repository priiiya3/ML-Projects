import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CameraCalibration():
    """ This class calibrates camera. We are using chessboard here for camera calibration.

    Attributes:
        mtx (np.array): Camera matrix 
        dist (np.array): Distortion coefficients
    """

    def __init__(self, image_dir, nx, ny, debug=False):
        
        f_names = glob.glob("{}/*".format(image_dir))
        objectPoints = []
        imagePoints = []
        
        # Chessboard's corners coordinates.
        objCoordinates = np.zeros((nx*ny, 3), np.float32)
        objCoordinates[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # Accessing all sample images.
        for f in f_names:
            img = mpimg.imread(f)

            # Converting to grayscale form.
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Getting the chessboard's corner points.
            success, corners = cv2.findChessboardCorners(img, (nx, ny))
            if success:
                imagePoints.append(corners)
                objectPoints.append(objCoordinates)

        imageShape = (img.shape[1], img.shape[0])
        success, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objectPoints, imagePoints, imageShape, None, None)

        if not success:
            raise Exception("Unable to calibrate the Camera")

    def undistort(self, img):
        """ Returns a numpy array of undistorted image."""

        # Convert to grayscale form.
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
