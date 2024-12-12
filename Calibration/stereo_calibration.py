import numpy as np
import cv2 as cv
import glob
import os


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

dir = os.path.dirname(os.path.realpath(__file__))
chessboardSize = (9,6)
frameSize = (1920,1080)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)

objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
print(objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


PATH = '/Users/ananyarajan/Desktop/Instrument_Tracking/Calibration_Images'
Left = sorted(glob.glob(f'{PATH}/Left/*.png'))
Right = sorted(glob.glob(f'{PATH}/Right/*.png'))


for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    print(imgLeft)
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    # if retL and retR == True:
    if retL and retR and len(cornersL)==len(cornersR) and len(cornersL) == 9*6 and len(cornersR) == 9*6:

        objpoints.append(objp)
    

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

# undistort
dst = cv.undistort(imgL, cameraMatrixL, distL, None, newCameraMatrixL)

########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

########## Stereo Rectification #################################################

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

# path = r'C:\Users\tsuid\OneDrive\Documents\Python Scripts\ComputerVision-master\StereoVisionDepthEstimation\calibrationCoefficients.yaml'
def save_coefficients(mtxL, distL, mtxR, distR, path, rot, trans, Q):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)

    # Left
    cv_file.write("KL", mtxL)
    cv_file.write("DL", distL)

    # Right
    cv_file.write("KR", mtxR)
    cv_file.write("DR", distR)

    cv_file.write("rot", rot)
    cv_file.write("trans", trans)

    # Map
    cv_file.write('sLX',stereoMapL[0])
    cv_file.write('sLY',stereoMapL[1])
    cv_file.write('sRX',stereoMapR[0])
    cv_file.write('sRY',stereoMapR[1])

    cv_file.write('Q', Q)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()



path = '/Users/ananyarajan/Desktop/Instrument_Tracking/StereoMap.yaml'
print("Saving parameters!")
save_coefficients(newCameraMatrixL, distL, newCameraMatrixR, distR, path, rot, trans, Q)
print("Camera L matrix : \n")
print(newCameraMatrixL)
print("Camera R Matrix : \n")
print(newCameraMatrixR)
print("Dist L Matrix : \n")
print(distL)
print("Dist R Matrix : \n")
print(distR)
print("Q : \n")
print(Q)
