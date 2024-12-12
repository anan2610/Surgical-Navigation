
# Calibration - Python

Calibration here is performed to calibrate the stereoscopic camera (a system that uses 2 cameras).
In the folder ‘Calibration_Images’, you will find pairs of images of a 9x7 chessboard captured using the stereo camera. 

Run stereo_calibration.py

Output is stored in StereoMap.yaml

# Marker Tracking - Python

‘calibration.py’, which uses StereoMap.yaml as its input, is imported into ‘disparity_pixel.py’
The second input is the left and right videos of the moving marker. Here, it is a video of the robotic arm moving a 4x4 ArUco marker in a circle – you will find the videos in the folder ‘Videos’. The circle has a slight inclination, meaning that the camera will also be capturing depth, apart from the X and Y coordinates. 

Run disparity_pixel.py

Output is stored in positions.csv

# Visualizing Data – Matlab

‘position.m’ will plot experimental data against ground truth data. Some adjustments are made to the data to set them in the same plane as the ground truth data.
The input to this is ‘positions.csv’.

Run position.m
Output is a set of graphs and the RMSE calculated between experimental and actual data



