""""Modle to detect Road Lanes Line."""


import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

class FindLaneLines:
    """ Parameters Tunning."""
    def __init__(self):
        
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLinesDetection()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def imageProcessing(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def videoProcessing(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)



def main():
    findLaneLines = FindLaneLines()

    video = 1

    # for lane detection in "Video" set video to 1.
    if video:
        findLaneLines.videoProcessing("project_video.mp4","testVideo_output.mp4")
    
    # for lane detection in "Image" set video to 0.
    else:
        findLaneLines.imageProcessing("curve_image.jpg", "curvedLine_image.jpg")

if __name__ == "__main__":
    main()