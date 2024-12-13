% Base code from https://github.com/AngeloDamante/arm-manipulator-5dof

%% import the Robot and add the Virtual End Effector
clear
clc
% robot = importrobot('RobotModel/tinkerkit4DOF.urdf');
robot2 = importrobot('RobotModel/tinkerkit.urdf');
numJoints = numel(homeConfiguration(robot2));
axes = show(robot2);
% axes.CameraPosition = [5 5 5];
axes.CameraPositionMode = 'auto';
%%
eeOffset=0.16; % gripper position - specifies the distance from the last joint
% to the gripper along the Z-axis
eeBody=robotics.RigidBody('end_effector');
setFixedTransform(eeBody.Joint, trvec2tform([0 0 eeOffset]));
addBody(robot2,eeBody,'link4'); %adding a link eeBody to link4 (parent),
% we are telling which point we want the IK solver
% to solve for the cartesian plane
clc
%% Home position
% config = homeConfiguration(robot);
config2 = homeConfiguration(robot2);
config2(1).JointPosition=pi/2;
config2(2).JointPosition=pi/2;
config2(3).JointPosition=pi/4;
config2(4).JointPosition=pi/4;
config2(5).JointPosition=pi/2;
show(robot2,config2);
title('Braccio: HOME Configuration')
axis([-0.3 0.3 -0.3 0.3 -0.1 0.5]);
hold on


% Linear without depth - used for calibration

% 1 unit in robot = real world 0.08104 cm (calculated on Feb 22 2024)
% 1 unit in camera coordinates = real world 0.75459 cm (calculated on Feb 22 2024)  
% 1 cm in real world = 12.34 robot units
% 1 cm in real world = 1.325 camera units

%1 cm in real world depth = 21.402 camera units

% 1 unit in waypoints = 1069.316 units in robot
% 1 unit in robot = 0.00093 units of waypoints

% point1=[0.1 0.1 0.15]; %config=[pi/2 pi/2 pi/4 pi/4]
% point2 = [0.1 0.15 0.15];
% point3=[0.1 0.25 0.15];
% point4 = [0 0.1 0.15];
% wayPoints=[point1; point2; point3; point2; point1];

% Linear 
% point1=[0.1 0.1 0.1]; %config=[pi/2 pi/2 pi/4 pi/4]
% point2 = [0.1 0.15 0.15];
% point3=[0.1 0.2 0.2];
% point4 = [0 0.1 0.1];
% wayPoints=[point1; point2; point3; point2; point1];


% 3D Parabola
% point1=[0.03 0.12 0.2]; %config=[pi/2 pi/2 pi/4 pi/4]
% point2 = [0.06 0.08 0.14];
% point3=[0.03 0.12 0.08];
% 
% wayPoints=[point1; point2; point3; point2; point1];

% 2D Parabola
% point1=[0.1 0.08 0.2]; %config=[pi/2 pi/2 pi/4 pi/4]
% point2 = [0.1 0.09263 0.1699];
% point2_5 = [0.1 0.09895,0.1575];
% point3 = [0.1 0.1116 0.138];
% point4 = [0.1 0.12 0.128889];
% point5 = [0.1 0.14 0.12];
% point6 = [0.1 0.16 0.128889];
% point7 = [0.1 0.1684 0.138];
% point7_5 =[0.1 0.1811 0.1575];
% point8 = [0.1 0.1874 0.1699];
% point9 =[0.1 0.2 0.2];
% wayPoints = [point1; point5; point9];
% wayPoints=[point1; point2; point2_5; point3; point4; point5; point6; point7; point7_5; point8; point9; point8; point7_5; point7; point6; point5; point4; point3; point2_5; point2; point1];

% Circle
% point1=[0.1 0.2 0.1];
% point2 = [0.1 0.2 0.2];
% point3=[0.1 0.1 0.2];
% point4 = [0.1 0.1 0.1];
% wayPoints=[point1; point2; point3; point4; point1];

% 3D'S'
% point1=[0 0.12 0.12]; %config=[pi/2 pi/2 pi/4 pi/4]
% point2 = [0.04 0.17 0.07];
% point3 = [0.08 0.22 0.12];
% point4 = [0.10 0.17 0.17];
% point5=[0.12 0.12 0.22];
% point6=[0.16 0.17 0.27];
% point7 = [0.20 0.22 0.22];
% wayPoints=[point1; point2; point3; point4; point5; point6; point7; point6; point5; point4; point3; point2; point1];%; point5; point6; point7; point8; point7; point6; point5];

% Squiggly
% point1=[0.1 0.1 0.1]; %config=[pi/2 pi/2 pi/4 pi/4]
% point2 = [0.14 0.2 0.14];
% point3=[0.17 0.1 0.17];
% point4 = [0.2 0.2 0.2];
% wayPoints=[point1; point2; point3; point4; point3; point2; point1];%; point5; point6; point7; point8; point7; point6; point5];

% 3D circle (inclined circle)

% point1 = [0.1 0.1 0.2];
% point2 = [0.0792 0.15 0.23];
% point3 = [0.1 0.2 0.2];
% point4 = [0.15 0.2207 0.17];
% point5 = [0.2 0.2 0.14];
% point6 = [0.2207 0.15 0.11];
% point7 = [0.2 0.1 0.14];
% point8 = [0.15 0.0792 0.17];
% wayPoints = [point1; point2; point3; point4; point5; point6; point7; point8; point1]; %point2; point3; point4; point5; point6; point7; point8; point1];

% 3D circle with Y shifted (inclined circle)
% point1 = [0.1 0.12 0.2];
% point2 = [0.0792 0.17 0.23];
% point3 = [0.1 0.22 0.2];
% point4 = [0.15 0.2407 0.17];
% point5 = [0.2 0.22 0.14];
% point6 = [0.2207 0.17 0.11];
% point7 = [0.2 0.12 0.14];
% point8 = [0.15 0.0992 0.17];
% wayPoints = [point1; point2; point3; point4; point5; point6; point7; point8; point1]; %point2; point3; point4; point5; point6; point7; point8; point1];

% 3D circle with Z and Y shifted (inclined circle)
point1 = [0.1 0.12 0.22];

point3 = [0.1 0.22 0.22];

point5 = [0.2 0.22 0.16];

point7 = [0.2 0.12 0.16];
wayPoints = [point1; point3; point5; point7; point1];% point7; point8; point1];

% point2 = [0.0792 0.17 0.25];
% point4 = [0.15 0.2407 0.19];
% point6 = [0.2207 0.17 0.13];
% point8 = [0.15 0.0992 0.19];
% wayPoints = [point1; point2; point3; point4; point5; point6; point7; point8; point1]; %

% 2D circle (same as previous but no inclination)

% point1 = [0.1 0.1 0.2];
% point2 = [0.1 0.15 0.23];
% point3 = [0.1 0.2 0.2];
% point4 = [0.1 0.2207 0.17];
% point5 = [0.1 0.2 0.14];
% point6 = [0.1 0.15 0.11];
% point7 = [0.1 0.1 0.14];
% point8 = [0.1 0.0792 0.17];
% wayPoints = [ point2;  point4;  point6;  point8; point2]; %point2; point3; point4; point5; point6; point7; point8; point1];

% 3D Infinity
% point1=[0.08 0.22 0.08];
% point2 = [0.22 0.22 0.22];
% point3=[0.08 0.08 0.22];
% point4 = [0.22 0.08 0.08];
% point5 = [0.2 0.2 0.15];
% wayPoints=[point1; point2; point4; point3; point1];

% 2D infinity
% point1=[0.08 0.22 0.08];
% point2 = [0.08 0.22 0.22];
% point3=[0.08 0.08 0.08];
% point4 = [0.08 0.08 0.22];
% wayPoints=[point1; point2; point3; point4; point1];

% 2D infinity - new
% point1=[0.08 0.22 0.15];
% point2 = [0.08 0.22 0.24];
% point3=[0.08 0.12 0.15];
% point4 = [0.08 0.12 0.24];
% wayPoints=[point1; point2; point3; point4; point1];

% 2D infinity - new 2
% point1=[0.08 0.30 0.20];
% point2 = [0.08 0.30 0.26];
% point3=[0.08 0.20 0.20];
% point4 = [0.08 0.20 0.26];
% wayPoints=[point1; point2; point3; point4; point1];

plot3(wayPoints(:,1),wayPoints(:,2),wayPoints(:,3),'.','MarkerSize',40, 'MarkerEdgeColor','k'); %passage points of EE
hold on
traj=cscvn(wayPoints');
fnplt(traj,'r',2);
grid on
hold off
%% Inverse Kinematics Trajectory
ik = robotics.InverseKinematics('RigidBodyTree',robot2);
ik.SolverAlgorithm = 'LevenbergMarquardt';
weights = [0 0 0 1 1 1]; % first three weights are the first three rotations in xyz
% last three are the weights for the three translations
% in xyz
initialguess = config2;
[n,~]=size(wayPoints);
totalPoints=n*30;
% totalPoints = 500;
x=linspace(0,traj.breaks(end),totalPoints);
eePos=ppval(traj,x);
for idx = 1:size(eePos,2)
    tform = trvec2tform(eePos(:,idx)');
    configSoln(idx,:) = ik('end_effector',tform,weights,initialguess);
    initialguess = configSoln(idx,:);
end
%% Trajectory Tracking
figure
title('Waypoints Tracking');
config2(5).JointPosition = 0;
config2(6).JointPosition=85*pi/180;
for idx = 1:size(eePos,2)
    config2(1).JointPosition=configSoln(idx,1).JointPosition;
    config2(2).JointPosition=configSoln(idx,2).JointPosition;
    config2(3).JointPosition=configSoln(idx,3).JointPosition;
    config2(4).JointPosition=configSoln(idx,4).JointPosition;
%     if  rem(idx,10) == 0
%         config2(5).JointPosition=idx; % wrist rotates 10 degrees at every 10th point
%     else 
%         config2(5).JointPosition;
%     end
    show(robot2,config2, 'PreservePlot', false,'Frames','off');
    hold on
    if  idx==1
        fnplt(traj,'r',2);
        plot3(wayPoints(:,1),wayPoints(:,2),wayPoints(:,3),'.','MarkerSize',40, 'MarkerEdgeColor','k');
    end
    pause(0.01)
end
hold off
%% Matrix of Joint Commands
JointCommandsRad=zeros(size(eePos,2),numJoints);
wayPoints=wayPoints';
for i = 1:size(eePos,2)
    JointCommandsRad(i,1)=configSoln(i,1).JointPosition;
    JointCommandsRad(i,2)=configSoln(i,2).JointPosition;
    JointCommandsRad(i,3)=configSoln(i,3).JointPosition;
    JointCommandsRad(i,4)=configSoln(i,4).JointPosition;
%     if  rem(i,10) == 0
%         JointCommandsRad(i,5)=i*pi/180;
%     end
end
JointCommandsRad=[JointCommandsRad(1,:); JointCommandsRad];
JointCommandsDeg=JointCommandsRad*180/pi;
%% Joint Commands Signal
% tot=20;
% step=tot/totalPoints;
% time=0:step:tot;
% figure
% base.time=time';
% base.signals.values=JointCommandsDeg(:,1);
% subplot(2,3,1);
% plot(time,JointCommandsDeg(:,1)');
% xlabel('Time');
% ylabel('Degrees');
% title('Base Motor Signal')
% grid on
% shoulder.time=time';
% shoulder.signals.values=JointCommandsDeg(:,2);
% subplot(2,3,2);
% plot(time,JointCommandsDeg(:,2)');
% xlabel('Time');
% ylabel('Degrees');
% title('Shoulder Motor Signal')
% grid on
% elbow.time=time';
% elbow.signals.values=JointCommandsDeg(:,3);
% subplot(2,3,3);
% plot(time,JointCommandsDeg(:,3)');
% xlabel('Time');
% ylabel('Degrees');
% title('Elbow Motor Signal')
% grid on
% wrist.time=time';
% wrist.signals.values=JointCommandsDeg(:,4);
% subplot(2,3,4);
% plot(time,JointCommandsDeg(:,4)');
% title('Wrist Motor Signal')
% grid on

filePath = '3D_Circle_New_angles4.xlsx';
columnNames = {'Base', 'Shoulder', 'Elbow', 'Wrist'};
JCD = table(JointCommandsDeg(:,1), JointCommandsDeg(:,2), JointCommandsDeg(:,3), JointCommandsDeg(:,4), 'VariableNames', columnNames);
writetable(JCD,filePath);


% wrist_roll.time=time';
% wrist_roll.signals.values=JointCommandsDeg(:,5);
% subplot(2,3,5);
% plot(time,JointCommandsDeg(:,5)');
% title('Wrist Rot Signal')
% grid on

%% Kinematics
% DH parameters from https://github.com/AngeloDamante/arm-manipulator-5dof

dh_params = [[0.0, 1.5708, 70, -1.5708]; 
    [125.0, 0.0, 0.0, 0.0];
    [125.0, 0.0, 0.0, -1.5708]; 
    [185.0, 0.0, 0.0,-1.5708];
    [0.0, 1.5708, 0.0, 0.0]]; % adA, adAlpha, adD, adOffset

% dh_params = [[0.0, -1.5708, -71.5, 0]; 
%     [-125.0, 0.0, 0.0, 1.5708];
%     [-125.0, 0.0, 0.0, 0]; 
%     [0, 1.5708, 0.0,-1.5708];
%     [0.0, 0, 192, 0.0]]; % adA, adAlpha, adD, adOffset

joint_angles = JointCommandsRad;

%Initialize transformation matrix

T = eye(4);
end_effector_positions = zeros(100, 3);
for i = 1:size(joint_angles, 1)
    T = eye(4); % Reset transformation matrix for each position
    
    % Calculate transformation matrix for each joint and multiply them
    for j = 1:size(dh_params, 1)
      
        theta = joint_angles(i, j);
        d = dh_params(j, 3);
        a = dh_params(j, 1);
        alpha = dh_params(j, 2);
        offset = dh_params(j,4);
        transformation = [
            cos(theta+offset), -sin(theta+offset)*cos(alpha), sin(theta+offset)*sin(alpha), a*cos(theta+offset);
            sin(theta+offset), cos(theta+offset)*cos(alpha), -cos(theta+offset)*sin(alpha), a*sin(theta+offset);
            0, sin(alpha), cos(alpha), d;
            0, 0, 0, 1
        ];
        
        T = T * transformation;
    end
    
    % Translation matrix
    end_effector_positions(i, :) = T(1:3, 4)';
    % Rotation matrix
    end_effector_orientations{i} = T(1:3, 1:3);
end

figure;
scatter3(end_effector_positions(:, 1), end_effector_positions(:, 2),end_effector_positions(:, 3), 'filled');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End Effector Positions');
grid on;

filePath = 'End_Effector_Positions_Circle5.xlsx';
columnNames = {'X', 'Y', 'Z'};
JCD = table(end_effector_positions(:, 1), end_effector_positions(:, 2), end_effector_positions(:, 3), 'VariableNames', columnNames);
writetable(JCD,filePath);




