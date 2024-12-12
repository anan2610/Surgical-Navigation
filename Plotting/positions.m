clear; close all;
set(0, 'DefaultFigureVisible', 'on');
%% Robot circle actual positions file

data1 = readtable('/Users/ananyarajan/Documents/GitHub/PainNavigation/Instrument_Tracking/Instrument_Position_Tracking/positions22.csv'); % tracking data ~ experimental data
data2 = readtable('/Users/ananyarajan/Desktop/Robot/End_Effector_Positions_Circle5.xlsx'); % ground truth data

% data1 = readtable('/Users/ananyarajan/Documents/GitHub/PainNavigation/Instrument_Tracking/Instrument_Position_Tracking/Robot_Video/Circle/Circle_4_13_24/Circle1.xlsx');
% data1 = readtable('/Users/ananyarajan/Documents/GitHub/PainNavigation/Instrument_Tracking/Instrument_Position_Tracking/Robot_Video/Robot_Videos_Old/Misc/C7.xlsx');
% data2 = readtable('/Users/ananyarajan/Documents/GitHub/PainNavigation/Instrument_Tracking/Instrument_Position_Tracking/End_Effector_Positions_3D_S.xlsx');
data1 = table2array(data1(10:122,:)); 
data2 = table2array(data2);

CF_X = 1.31; % camera to real world X axis
CF_Y = 1.33; % camera to real world Y axis
CF_Z = 1.89; % camera to real world depth 
CF_R = 0.074614541; % robot to real-world


%Changing camera coordinates to real world coordinates
l1 = length(data1(:,1));
theta_radz = 0*pi/180;
theta_radx = 0*pi/180;
theta_rad = 0*pi/180;

Rx = [1 0 0; 0 cos(theta_radx) -sin(theta_radx); 0 sin(theta_radx) cos(theta_radx)]; %rotating about x-axis
Ry = [cos(theta_rad) 0 sin(theta_rad); 0 1 0; -sin(theta_rad) 0 cos(theta_rad)]; 
Rz = [cos(theta_radz) -sin(theta_radz) 0; sin(theta_radz) cos(theta_radz) 0; 0 0 1]; %rotating about z axis

x1 = CF_X*data1(:,3); %5,3,4
y1 = CF_Y*data1(:,4);
z1 = CF_Z*data1(:,5);

% Averaging

% for i = 1:l1-1
%     x1(i) = (x1(i) + x1(i+1))/2;
%     y1(i) = (y1(i) + y1(i+1))/2;
%     z1(i) = (z1(i) + z1(i+1))/2;
% end


% rotated_points = -Rx * Ry * Rz * [x1'; y1'; z1']; % rotating and inverting

%  Apply transformation to real-world coordinates
% x1 = rotated_points(1,:)';
% y1 = rotated_points(2,:)';
% z1 = rotated_points(3,:)';


%Changing robot coordinates to real world coordinates
l2 = length(data2);
x2 = CF_R*data2(:,3);
y2 = CF_R*data2(:,1);
z2 = CF_R*data2(:,2);


figure(1);
plot3(x1,y1,z1,'.');
title('Experimental Data from Camera', 'FontSize', 15, 'FontName', 'Times New Roman');
xlabel('X', 'FontSize', 15, 'FontName', 'Times New Roman');
ylabel('Y', 'FontSize', 15, 'FontName', 'Times New Roman');
zlabel('Z', 'FontSize', 15, 'FontName', 'Times New Roman');

figure(2);
plot3(x2, y2, z2,'.');
title('Theoretical Data (Ground Truth)', 'FontSize', 15, 'FontName', 'Times New Roman');
xlabel('X', 'FontSize', 15, 'FontName', 'Times New Roman');
ylabel('Y', 'FontSize', 15, 'FontName', 'Times New Roman');
zlabel('Z', 'FontSize', 15, 'FontName', 'Times New Roman');

figure(3);
plot3(x1, y1, z1,'.', x2, y2, z2,'.');
title('Theoretical Pattern vs. Experimental Pattern before Processing', 'FontSize', 15, 'FontName', 'Times New Roman');
xlabel('X');
ylabel('Y');
zlabel('Z');
legend('Experimental Data', 'Theoretical Data');

%% Remove outliers
% indices =  z1<14;
% 
% % Remove the corresponding rows from data1
% data1(indices, :) = [];
% 
% x1(indices) = [];
% y1(indices) = [];
% z1(indices) = [];
% data1(:,3) = x1;
% data1(:,4) = y1;
% data1(:,5) = z1;
% 
% % Averaging
% 
% for i = 1:length(data1)-1
%     x1(i) = (x1(i) + x1(i+1))/2;
%     y1(i) = (y1(i) + y1(i+1))/2;
%     z1(i) = (z1(i) + z1(i+1))/2;
% end
% 
% data1(:,3) = x1;
% data1(:,4) = y1;
% data1(:,5) = z1;

%% Interpolation
% Upsampling

num_points_set1 = length(data1);
num_points_set2 = length(data2);
upsample_ratio = num_points_set2 / num_points_set1; % when data1 < data2
new_data1 = interp1(1:num_points_set1, data1, linspace(1, num_points_set1, num_points_set2));

% Downsampling

% num_points_set1 = length(data1);
% num_points_set2 = length(data2);
% upsample_ratio = num_points_set1 / num_points_set2; % when data2 < data1
% new_data2 = interp1(1:num_points_set2, data2, linspace(1, num_points_set2, num_points_set1));


x1 = CF_X*new_data1(:,3); 
y1 = CF_Y*new_data1(:,4);
z1 = CF_Z*new_data1(:,5);

% Rotating with previous values

rotated_points = -Rx * Ry * Rz * [x1'; y1'; z1']; % rotating and inverting

x1 = rotated_points(1,:)';
y1 = rotated_points(2,:)';
z1 = rotated_points(3,:)';
x1 = smoothdata(x1, 'lowess');
y1 = smoothdata(y1, 'lowess');
z1 = smoothdata(z1, 'lowess');

% figure(4);
% hold on;
% plot3(x1(1:45),y1(1:45),z1(1:45),'r.', x2(1:45), y2(1:45), z2(1:45),'r.');
% plot3(x1(46:90),y1(46:90),z1(46:90),'g.', x2(46:90), y2(46:90), z2(46:90),'g.');
% plot3(x1(91:135),y1(91:135),z1(91:135),'b.', x2(91:135), y2(91:135), z2(91:135),'b.');
% plot3(x1(136:end),y1(136:end),z1(136:end),'m.', x2(136:end), y2(136:end), z2(136:end),'m.');
% legend('camera', 'robot');
% xlabel('x');
% ylabel('y');
% zlabel('z');
% title('robot vs. camera');
% hold off;

figure(5);
title('Experimental X, Y and Z values vs. Time before SVD', 'FontSize', 15, 'FontName', 'Times New Roman'); %after filtering
title('Theoretical X, Y and Z values vs. Time before SVD', 'FontSize', 15, 'FontName', 'Times New Roman'); %after filtering
subplot(2,3,1); plot(x1); title('X', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,2); plot(y1); title('Y', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,3); plot(z1); title('Z', 'FontSize', 15, 'FontName', 'Times New Roman'); 
subplot(2,3,4); plot(x2); title('X', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,5); plot(y2); title('Y', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,6); plot(z2); title('Z', 'FontSize', 15, 'FontName', 'Times New Roman');

figure(10);
hold on;
plot3(x1, y1, z1);
plot3(x2, y2, z2);
title('Patterns Before SVD', 'FontSize', 15, 'FontName', 'Times New Roman');
xlabel('X', 'FontSize', 15, 'FontName', 'Times New Roman'); 
ylabel('Y', 'FontSize', 15, 'FontName', 'Times New Roman'); 
zlabel('Z', 'FontSize', 15, 'FontName', 'Times New Roman');
legend('Experimental', 'Theoretical');
hold off;

C(:,1) = x1;
C(:,2) = y1;
C(:,3) = z1;
D(:,1) = x2;
D(:,2) = y2;
D(:,3) = z2;
C = reshape(C.',1,[]);
D = reshape(D.',1,[]);
rho1 = corr(C',D'); % spearman

figure(6); 
plot(C', D'); 
title('Correlation before SVD', 'FontSize', 15, 'FontName', 'Times New Roman');
xlabel('Theoretical Data', 'FontSize', 15, 'FontName', 'Times New Roman');
ylabel('Experimental Data', 'FontSize', 15, 'FontName', 'Times New Roman');

% Beginning SVD

A(:,1) = y1;
A(:,2) = z1;
A(:,3) = x1;
B(:,1) = x2;
B(:,2) = y2;
B(:,3) = z2;

n = length(A(:,1));

% recover the transformation
[Rc, tc] = euclidean_transform_3D(A, B);

A_transformed = (Rc*A' + repmat(tc, 1, n))';

figure(7);
title('Experimental X, Y and Z values vs. Time after SVD', 'FontSize', 15, 'FontName', 'Times New Roman');
title('Theoretical X, Y and Z values vs. Time after SVD', 'FontSize', 15, 'FontName', 'Times New Roman');

subplot(2,3,1); plot(A_transformed(:,1)); title('X', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,2); plot(A_transformed(:,2)); title('Y', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,3); plot(A_transformed(:,3)); title('Z', 'FontSize', 15, 'FontName', 'Times New Roman'); 
subplot(2,3,4); plot(x2); title('X', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,5); plot(y2); title('Y', 'FontSize', 15, 'FontName', 'Times New Roman'); subplot(2,3,6); plot(z2); title('Z', 'FontSize', 15, 'FontName', 'Times New Roman');


figure(8);
hold on;
plot3(B(:,1), B(:,2), B(:,3));
plot3(A_transformed(:,1),A_transformed(:,2),A_transformed(:,3));
title('Patterns after SVD', 'FontSize', 15, 'FontName', 'Times New Roman');
xlabel('X', 'FontSize', 15, 'FontName', 'Times New Roman'); 
ylabel('Y', 'FontSize', 15, 'FontName', 'Times New Roman'); 
zlabel('Z', 'FontSize', 15, 'FontName', 'Times New Roman');
legend('Theoretical', 'Experimental');
hold off;

% Find the error
rmse = sqrt(mean((A_transformed - B).^2));
disp(['RMSE:', num2str(rmse)]);
mean(rmse)

A_reshaped = reshape(A_transformed.',1,[]);
B_reshaped = reshape(B.',1,[]);
rho2 = corr(A_reshaped',B_reshaped') % spearman

figure(9); 
plot(A_reshaped', B_reshaped'); 
title('Correlation after SVD', 'FontSize', 15, 'FontName', 'Times New Roman');
xlabel('Theoretical Data','FontSize', 15, 'FontName', 'Times New Roman');
ylabel('Experimental Data', 'FontSize', 15, 'FontName', 'Times New Roman');

%% RMSE
% x3 = -CF_R*new_data1(:,1);%+36; 
% y3 = -CF_R*new_data1(:,2);%+45;
% z3 = -CF_R*new_data1(:,3);%-17;
% 
% squared_e_x = (x3 - x1).^2; 
% squared_e_y = (y3 - y1).^2;
% squared_e_z = (z3 - z1).^2;
% 
% mse = mean(squared_e_x + squared_e_y + squared_e_z);
% 
% rmse = sqrt(mse)
% 
% rmse_x = sqrt(squared_e_x);
% rmse_y = sqrt(squared_e_y);
% rmse_z = sqrt(squared_e_z);
% 
% avg_rmse_x = sqrt(mean(squared_e_x))
% avg_rmse_y = sqrt(mean(squared_e_y))
% avg_rmse_z = sqrt(mean(squared_e_z))
% 
% rmse_avg = mean(avg_rmse_x+ avg_rmse_y + avg_rmse_z);
% 
% avg_mse_x = mean(squared_e_x);
% avg_mse_y = mean(squared_e_y);
% avg_mse_z = mean(squared_e_z);
% 
% n = size(squared_e_x, 1);  % Get the number of elements in squared_e_x(:,1)
% abs_mse_x = zeros(n, 1);     % Initialize a vector to store the results
% abs_mse_y = zeros(n, 1);
% abs_mse_z = zeros(n, 1);
% for i = 1:n
%     abs_mse_x(i) = abs(squared_e_x(i,1) - avg_mse_x);
%     abs_mse_y(i) = abs(squared_e_y(i,1) - avg_mse_y);
%     abs_mse_z(i) = abs(squared_e_z(i,1) - avg_mse_z);
% end
% 
% n = size(squared_e_x, 1);  % Get the number of elements in squared_e_x(:,1)
% abs_rmse_x = zeros(n, 1);     % Initialize a vector to store the results
% abs_rmse_y = zeros(n, 1);
% abs_rmse_z = zeros(n, 1);
% 
% for i = 1:n
%     abs_rmse_x(i) = abs(rmse_x(i,1) - avg_rmse_x);
%     abs_rmse_y(i) = abs(rmse_y(i,1) - avg_rmse_y);
%     abs_rmse_z(i) = abs(rmse_z(i,1) - avg_rmse_z);
% end
% 
% mean(abs_rmse_x)
% mean(abs_rmse_y)
% mean(abs_rmse_z)

%%
% abs_rmse_y = mean(abs_rmse_y)
%Averaging
% x1 = zeros(1,l);
% y1 = zeros(1,l);
% z1 = zeros(1,l);
% 
% for i = 1:l-1
%     x1(i) = (x(i) + x(i+1))/2;
%     y1(i) = (y(i) + y(i+1))/2;
%     z1(i) = (z(i) + z(i+1))/2;
% end
% 
% figure(2);
% plot3(x1,y1,z1,'.');
% xlabel('x'); ylabel('y'); zlabel('z');
% title('after averaging');

% X vs. Y
% figure(3);
% plot(y,x);
% xlabel('x1');
% ylabel('y1');
% title('x vs. y (averaged)');

% figure(3);
% plot3(z1, x1, y1,'.',x2, y2, z2, 'p');
% title ('Original vs. estimated');
% legend('Estimated','Original');

% for i = 1:length(x)
% plot3(x(1:i),y(1:i),z(1:i),'.');
%     drawnow;
% end
% print(x1)

% indices = (z>5);
% x(indices) = [];
% y(indices) = [];
% z(indices) = [];
% 
% indices = (z > 0) & (z < 50);
% x(indices) = x(indices)*1e8;
% y(indices) = y(indices)*1e8;
% z(indices) = z(indices)*1e8;

% Draw Original Camera Coordinates

% figure;
% plot3(x,y,z,'.');
% z = z*100;
% figure;
% for i = 1:length(x)
%     
%     plot3(x(1:i),y(1:i),z(1:i),'.');
%     drawnow;
% end
% xlim([0 125])

function [R, t] = euclidean_transform_3D(A, B)
    % A,B - Nx3 matrix
    % return:
    %     R - 3x3 rotation matrix
    %     t = 3x1 column vector
    
    assert(size(A, 1) == size(B, 1));

    % number of points
    N = size(A, 1);

    centroid_A = mean(A);
    centroid_B = mean(B);

    % centre matrices
    AA = A - repmat(centroid_A, N, 1);
    BB = B - repmat(centroid_B, N, 1);

    % covariance of datasets
    H = transpose(AA) * BB;

    % matrix decomposition on rotation, scaling, and rotation matrices
    [U, ~, Vt] = svd(H);

    % resulting rotation
    R = Vt * U';

    % handle svd sign problem
    if det(R) < 0
        Vt(3,:) = Vt(3,:);
        
        R = Vt * U';
    end

    t = -R*centroid_A' + centroid_B';

end