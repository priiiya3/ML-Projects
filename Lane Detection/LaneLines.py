import cv2
import numpy as np
import matplotlib.image as mpimg

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLinesDetection:
    """ Class containing information about detected lane lines.
    """
    def __init__(self):
        
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonZeroX = None
        self.nonZeroY = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('images/left.png')
        self.right_curve_img = mpimg.imread('images/right.png')
        self.keep_straight_img = mpimg.imread('images/strt.png')
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        ####### HYPER-PARAMETERS
        
        # Number of sliding windows
        self.numWindows = 9
        
        # Width of the the windows and its margin
        self.margin = 100

        # Mininum number of pixels found to recenter window
        self.minpix = 50



    def forward(self, img):
        """detects the lane lines in the image passed and returns image containing lane line details."""

        self.extract_features(img)
        
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Calculates all pixelS present in a specific window.

        Parameters passed:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window

        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """

        topLeft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        xCoord = (topLeft[0] <= self.nonZeroX) & (self.nonZeroX <= bottomright[0])
        yCoord = (topLeft[1] <= self.nonZeroY) & (self.nonZeroY <= bottomright[1])
        
        return self.nonZeroX[xCoord & yCoord], self.nonZeroY[xCoord & yCoord]


    def extract_features(self, img):
        """ To Extract  the features of the image passed in Binary form.
        Parameters passed:
            img (np.array): A binary image
        """

        self.img = img

        # Height of of windows (depends on number of windows(i.e, numWindows) and image shape)
        self.window_height = np.int(img.shape[0]//self.numWindows)

        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonZeroX = np.array(self.nonzero[1])
        self.nonZeroY = np.array(self.nonzero[0])

    def finding_lane_pixels(self, img):
        """Finds the pixels of the lane from a binary warped image passed.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            outputImage (np.array): An RGB image ( will be used to display result).
        """
        assert(len(img.shape) == 2)

        # Creating an output image for drawing and visualizing the result.
        outputImage = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        # lists to store pixels of left and right target lane.
        leftx, lefty, rightx, righty = [], [], [], []

        # Steping through the windows.:
        for i in range(self.numWindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Appending indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, outputImage

    def fit_poly(self, img):
        """ Detects the lane line in the image being passed and draw over it.

        Parameters:
            img (np.array): a binary warped image

        Returns:
            outputImage (np.array): RGB image with line(s) drawn over lane lines..
        """

        # Coordinates of left and right lane.
        leftx, lefty, rightx, righty, outputImage = self.finding_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        # Getting x and y values of image passed for plotting purpose.
        y_max = img.shape[0] - 1
        y_min = img.shape[0] // 3

        if len(lefty):
            y_max = max(y_max, np.max(lefty))
            y_min = min(y_min, np.min(lefty))

        if len(righty):
            y_max = max(y_max, np.max(righty))
            y_min = min(y_min, np.min(righty))

        ## Getting evenly spaced numbers
        ploty = np.linspace(y_min, y_max, img.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Visualizing the outputImage
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(outputImage, (l, y), (r, y), (135,206,235))

        leftR, rightR, pos = self.measure_curvature()

        return outputImage

    def plot(self, outputImage):
        np.set_printoptions(precision=6, suppress=True)

        leftR, rightR, pos = self.measure_curvature()

        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        
        if len(self.dir) > 10:
            self.dir.pop(0)

        Wth = 400
        Hth = 500
        widget = np.copy(outputImage[:Hth, :Wth])
        widget = widget // 2
        widget[0, :] = [0, 0, 255]
        widget[-1, :] = [0, 0, 255]
        widget[:, 0] = [0, 0, 255]
        widget[:, -1] = [0, 0, 255]
        outputImage[:Hth, :Wth] = widget

        direction = max(set(self.dir), key = self.dir.count)
        display_msg = "Keep Straight Ahead"

        curvature_msg = "Curvature = {:.0f} m".format(min(leftR, rightR))
        
        # if no curve ahead. (abs(value) <= 0.00015)
        if direction == 'F':
            y, x = self.keep_straight_img[:,:,3].nonzero()
            outputImage[y, x - 100 + Wth // 2] = self.keep_straight_img[y, x, :3]

        # if left curve ahead. (abs(value) < 0)
        if direction == 'L':
            y, x = self.left_curve_img[:, :, 3].nonzero()
            outputImage[y, x - 100 + Wth // 2] = self.left_curve_img[y, x, :3]
            display_msg = "Left Curve Ahead"

        # else if right curve ahead.
        if direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            outputImage[y, x - 100 + Wth // 2] = self.right_curve_img[y, x, :3]
            display_msg = "Right Curve Ahead"

        # Display the Message.
        cv2.putText(outputImage, display_msg, org=(10, 340), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        if direction in 'LR':
            cv2.putText(outputImage, curvature_msg, org=(10, 420), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

        return outputImage

    def measure_curvature(self):
        """Measures the Radius of Curvature of the Lane. """

        ymeasure = 30/720
        xmeasure = 3.7/700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ymeasure

        # Compute R_curve (radius of curvature)
        left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
        right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = ((1280 // 2) - (xl + xr) // 2) * xmeasure
        
        return left_curveR, right_curveR, pos 
