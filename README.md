
## Surgical Navigation and Optical Tracking

Surgical navigation combines medical imaging and visualization, providing surgeons with positional information about the patient and instruments on an auxiliary display. This allows operators to plan the trajectory of instruments in real-time as they penetrate the skin.  

State-of-the-art navigation systems incorporate invasive retro-reflective markers inserted into the skin of the patient that can be detected with the help of infrared cameras. This technique is highly expensive, with these systems costing as much as $200,000 - $500,000.

Optical tracking, a much cheaper alternative to infrared tracking, relies on visible light to be able to track specific objects or markers. These markers are to be stuck on the patient and the surgical instruments while cameras follow their movement. 

In this project, with two low-cost web cameras, a stereoscopic camera is calibrated and programmed to detect the 3D position of moving fiducial
ArUco markers, non-invasive tags that store positional information. 

To test the accuracy of the system, the markers were moved with a robot in 3D with the objective of tracking movements patients and instruments during surgery.

# Calibration - Python

Calibration is performed to calibrate the stereoscopic camera (a system that uses 2 cameras).
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

