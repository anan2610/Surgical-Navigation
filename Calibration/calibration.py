import numpy as np
import cv2
import os

# hi lol
def loadCoefficients():
    
    cv_file = cv2.FileStorage(r"/Users/ananyarajan/Desktop/Instrument_Tracking/StereoMap.yaml", cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrixL = cv_file.getNode("KL").mat()
    dist_matrixL = cv_file.getNode("DL").mat()
    camera_matrixR = cv_file.getNode("KR").mat()
    dist_matrixR = cv_file.getNode("DR").mat()
    rot = cv_file.getNode("rot").mat()
    trans = cv_file.getNode("trans").mat()
    leftMapX = cv_file.getNode("sLX").mat()
    leftMapY = cv_file.getNode("sLY").mat()
    rightMapX = cv_file.getNode("sRX").mat()
    rightMapY = cv_file.getNode("sRY").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [camera_matrixL, dist_matrixL, camera_matrixR, dist_matrixR, rot, trans, leftMapX, leftMapY, rightMapX, rightMapY, Q]
    

def undistortRectify(frameR, frameL, leftMapX, leftMapY, rightMapX, rightMapY):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, leftMapX, leftMapY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, rightMapX, rightMapY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    # undistortedL = cv2.remap(frameL, leftMapX, leftMapY, cv2.INTER_NEAREST)
    # undistortedR = cv2.remap(frameR, rightMapX, rightMapY, cv2.INTER_NEAREST)

    return undistortedR, undistortedL
