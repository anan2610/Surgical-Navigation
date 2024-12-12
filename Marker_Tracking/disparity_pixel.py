
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import cv2.aruco as aruco
import pandas as pd

# Function for stereo vision and depth estimation
# import tri and cali from lab computer
import calibration
import time

def get_frame_time(frame_number, fps):
    return frame_number / fps

# Open both videos
# check if right video looks like right and left looks like left
video_pathl = r'/Users/ananyarajan/Desktop/Instrument_Tracking/Circle/CircleL1.mp4'
video_pathr = r'/Users/ananyarajan/Desktop/Instrument_Tracking/Circle/CircleR1.mp4'
capl = cv2.VideoCapture(video_pathl)
capr = cv2.VideoCapture(video_pathr)

# Stereo vision setup parameters CHECK!
frame_rate = 20    #Camera frame rate (maximum at 120 fps) # specs: https://www.amazon.com/Logitech-C920x-Pro-HD-Webcam/dp/B085TFF7M1/ref=sr_1_3?crid=MAKRBME33R82&keywords=usb%2Bcamera&qid=1678156008&sprefix=usb%2Bcamer%2Caps%2C201&sr=8-3&th=1
B = 12               #Distance between the cameras [cm] (old = 7.25cm, new = 12cm)
f = 14              #Camera lense's focal length [mm]
alpha = 110        #Camera field of view in the horisontal plane [degrees]

data = {'Marker': [], 'X': [], 'Y': [], 'Z': [], 'Frame': []}
frame_count = 0
disparity_x = []  # List to store disparity values
disparity_y = []
depth = []
x_value = []
y_value =[]
z_value = []
timestamp_data = []  # List to store timestamps


mtxL, distL, mtxR, distR, rot, trans, leftMapX, leftMapY, rightMapX, rightMapY, Q = calibration.loadCoefficients()
video_length = int(capl.get(cv2.CAP_PROP_FRAME_COUNT))

# obtaining fps and time points of each frame
fps = capl.get(cv2.CAP_PROP_FPS)
print(f'fps is {fps}')
for i in range(video_length):

    succes_right, frame_right = capr.read()
    succes_left, frame_left = capl.read()
    #check if either frame is None
    if frame_right is None or frame_left is None:
        break

    # Apply rectification maps
    frame_right = cv2.remap(frame_right, rightMapX, rightMapY, cv2.INTER_LINEAR)
    frame_left = cv2.remap(frame_left, leftMapX, leftMapY, cv2.INTER_LINEAR)

    frame_number = int(capl.get(cv2.CAP_PROP_POS_FRAMES))

    # Get the time point of the current frame
    time_point = get_frame_time(frame_number, fps)
    # print(time_point)

################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left, leftMapX, leftMapY, rightMapX, rightMapY)
   
   # Convert focal length from [mm] to pixel
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape
    
    Ox = int(width_left/2)
    Oy = int(height_left/2)

    
    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')
    
########################################################################################

    frame_count += 1

    """
    Uncomment for HSL/HSV - Comment for RGB
    """
    # Show the frames
    # cv2.imshow("frame right", frame_right) 
    # cv2.imshow("frame left", frame_left)
    # succes_right, frame_right = capr.read()
    # succes_left, frame_left = capl.read()
    # # define range of white color in HSV
    # # change it according to your need !
    # lower_white = np.array([0,0,0])
    # upper_white = np.array([255,255,255])

    # hsv_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    # hsv_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    # # # Threshold the HSV image to get only white colors
    # mask_right = cv2.inRange(hsv_right, lower_white, upper_white)
    # frame_right = cv2.bitwise_and(frame_right,frame_right, mask= mask_right)
    # mask_left = cv2.inRange(hsv_left, lower_white, upper_white)
    # frame_left = cv2.bitwise_and(frame_left,frame_left, mask= mask_left)

    # """
    # Uncomment for HSL/HSV - Comment for RGB
    # """
    # bgr_r = cv2.cvtColor(frame_right, cv2.COLOR2BGR)
    # bgr_l = cv2.cvtColor(frame_left, cv2.COLOR_HLS2BGR)
    # gray_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2GRAY)  # Chanqge grayscale
    # gray_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2GRAY)  # Change grayscale
   
    """
    Uncomment for RGB - Comment for HSL/HSV
    """
    gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    gray_l = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    # Read frames from video capture sources
    # success_right, frame_right = capr.read()
    # success_left, frame_left = capl.read()

    # # Define range of white color in HLS
    # # Change it according to your need!
    # lower_white = np.array([50, 50, 200])
    # upper_white = np.array([255, 255, 255])

    # # Convert frames to HLS color space
    # hls_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2HLS)
    # hls_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2HLS)

    # # Threshold the HLS image to get only white colors
    # mask_right = cv2.inRange(hls_right, lower_white, upper_white)
    # masked_frame_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)

    # mask_left = cv2.inRange(hls_left, lower_white, upper_white)
    # masked_frame_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)
    # cv2.imshow("Masked frame right", masked_frame_right)
    # cv2.imshow("Masked frame left", masked_frame_left)

    # bgr_r = cv2.cvtColor(masked_frame_right, cv2.COLOR_HLS2BGR)
    # bgr_l = cv2.cvtColor(masked_frame_left, cv2.COLOR_HLS2BGR)
    # gray_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2GRAY)  # Chanqge grayscale
    # gray_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2GRAY)  # Change grayscale



    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)#Aruco marker size 4x4 (+ border)
    parameters =  aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # lists of ids and the corners beloning to each id
    corners_r, ids_r, rejected_img_points_r = detector.detectMarkers(gray_r)
                                                            #cameraMatrix=matrix_coefficients,
                                                            #distCoeff=distortion_coefficients)
                                                            #output coordinates of the 4 corners and the ID of the marker 
    corners_l, ids_l, rejected_img_points_l = detector.detectMarkers(gray_l)
                                                            #cameraMatrix=matrix_coefficients,
                                                            #distCoeff=distortion_coefficients)
                                                            #output coordinates of the 4 corners and the ID of the marker 

     
    if not succes_right or not succes_left:                    
        break

    else:
       
        start = time.time()
        
        ################## CALCULATING DEPTH #########################################################

        center_right = 0
        center_left = 0

  
        if (np.all(ids_r is not None) & np.all(ids_l is not None)).all():
        # if np.all(ids_r is not None) and np.all(ids_l is not None):  # If there are markers found by detector
            zipped_r = zip(ids_r, corners_r)
            zipped_l = zip(ids_l, corners_l )
            
            ids_r, corners_r = zip(*(sorted(zipped_r)))
            ids_l, corners_l = zip(*(sorted(zipped_l)))
            # right is 2
            # left is 1
            
            
            axis_r = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
            axis_l = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
           
            for i in range(0, len(ids_l)):  # Iterate in markers

                if len(ids_l) != len(ids_r):
                    continue
                else:
                    # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    rvec_r, tvec_r, markerPoints_r = aruco.estimatePoseSingleMarkers(corners_r[i], 0.04, mtxR, distR)
                    rvec_l, tvec_l, markerPoints_l = aruco.estimatePoseSingleMarkers(corners_l[i], 0.04, mtxL, distL)
                    #40mm = 0.04m
                    #pull out the rotation of the marker and the tvec is the center of the four corners
                    # print(tvec_r)
                    x_r = int(np.mean(corners_r[i][0][:,0]))
                    x_l = int(np.mean(corners_l[i][0][:,0]))
                    y_r = int(np.mean(corners_r[i][0][:,1]))
                    y_l = int(np.mean(corners_l[i][0][:,1]))

                    center_point_right = np.array([x_r, y_r])
                    center_point_left = np.array([x_l, y_l])
                    vl = center_point_left[1]
                    ul = center_point_left[0]
                    vr = center_point_right[1]
                    ur = center_point_right[0]

                    aruco.drawDetectedMarkers(frame_right, corners_r)  # Draw A square around the markers
                    aruco.drawDetectedMarkers(frame_left, corners_l)  # Draw A square around the markers
                    
                    # If no marker can be caught in one camera show text "TRACKING LOST"
                    if not np.all(ids_r is not None) or not np.all(ids_l is not None):
                        cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                        cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

                    else:
        
                        disparity = math.sqrt((center_point_right[0] - center_point_left[0])**2 + (center_point_left[1] - center_point_right[1])**2) 
                        
                        # disparity = center_point_right[0] - center_point_left[0]

                        # disparity in pixels
                        disparity_x.append(disparity)
                        # print(type(disparity))
                        # disparity_y.append(disparity_Y)
                        timestamp_data.append(time_point)

                        Depth = (f_pixel*B)/disparity #depth in [cm], (pixel*cm/pixel = cm)
                        # print(Depth)
                        # print(len(disparity_x))
                        z_value.append(Depth)
                        x_calculation = (B*(ul - Ox))/disparity  #(Depth * center_point_right[0])/f_pixel
                        y_calculation = (B*f_pixel*(vl - Oy))/(f_pixel*disparity)    #(Depth * center_point_right[1])/(f_pixel*disparity)
                        
                        x_value.append(x_calculation)
                        y_value.append(y_calculation)
                        
        cv2.imshow("frame right", frame_right) 
        cv2.imshow("frame left", frame_left)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        end = time.time()
        totalTime = end - start
# Create a DataFrame with disparity values and timestamps
df = pd.DataFrame({'Timestamp': timestamp_data, 'Disparity': disparity_x, 'X': x_value, 'Y': y_value, 'Z': z_value})

# File path for the new Excel file
file_path = '/Users/ananyarajan/Desktpop/Instrument_Tracking/positions.xlsx'
df.to_excel(file_path, index=False)
capr.release()
capl.release()

cv2.destroyAllWindows()
