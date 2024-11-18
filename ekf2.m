M = readtable("Data/Recording 7.csv"); %change this depending on which recording you are using

x_data = table2array(M(:,17)); %easting
y_data = table2array(M(:,18)); %northing
speed_data = table2array(M(:,19)); %sped value
theta_data = mod(table2array(M(:,15)),2*pi); %heading 
theta_dot_data = table2array(M(:,7)); %angular velocity
timestamps = table2array(M(:,1));

% EKF Parameters
deltaT = diff(timestamps); % the time between each timestamps (deltaT)
steps = length(timestamps); %number of steps

model = @(x, deltaT) [x(1) + deltaT * x(3) * cos(x(4)); % create a model simulating the EKF (F)
      x(2) + deltaT * x(3) * sin(x(4)) ; 
      x(3); 
      x(4) + x(5) * deltaT; 
      x(5)];

% State vector: [eastings, northings, speed, theta, theta_dot_data]
x = [x_data(1); y_data(1); speed_data(1); theta_data(1); theta_dot_data(1)]; 

% Initial covariance matrix
P = cov(x);

R_1 = [0.6864, 0.4793, 0, 0, 0;
      0.4793, 0.4683, 0, 0, 0;
      0, 0, 0.4366, 0, 0;
      0, 0, 0, 0.0011, 0;
      0, 0, 0, 0, 0.0035];
%calculated in another file

% Create an array to preallocate space for storing results
estimated_positions = zeros(steps, 2);

% Set initial position based on our data
estimated_positions(1,:) = [x_data(1), y_data(1)];


for i = 2:steps
 deltaT2 = deltaT(i-1); % deltaT calculated between timestamps
 x_prediction = model(x, deltaT2); % prediction step 1/2
 
 % Jacobian of the motion model (F matrix or the prediction matrix)
 F = eye(5);
 F(1, 3) = deltaT2 * cos(mod(x(4),2*pi));
 % F(1, 5) = - deltaT2 * x(3) * sin(mod(x(4),2*pi)); % a previous file gave
 % us a different prediction matrix for some reason. 
 F(2, 3) = deltaT2 * sin(mod(x(4),2*pi));
 % F(2, 5) = deltaT2 * x(3) * cos(mod(x(4),2*pi));
 F(4, 5) = deltaT2;
 F; %testing purposes

 Q = diag([(0.8285*deltaT2).^2, (0.6843*deltaT2).^2, (0.6608*deltaT2).^2, (0.0332*deltaT2).^2, (0.0592*deltaT2).^2]);
 % Th diagonals are basically the standard deviations multiplied by deltaT
 % of that timestamp and all of it squared. 


 % Update the covariance matrix
 P_prediction = F * P * F' + Q; % prediction step 2/2

 if ~isnan(x_data(i)) && ~isnan(y_data(i)) % Check for valid GNSS data
  z = [x_data(i); y_data(i); speed_data(i); mod(theta_data(i),2*pi); theta_dot_data(i)];
  H = eye(5); 
  R = R_1;
 else % no GNSS data
  z = [0; 0;speed_data(i); - mod(theta_data(i),2*pi) + pi /2; theta_dot_data(i)]; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  H = eye(5);
  H(1, 1) = 0;
  H(2, 2) = 0;
  R = R_1;
 end

 % Compute Kalman gain
 Y = (z - H * x_prediction);
 S = (H * P * H' + R);
 K = P * H' / S; % kalman gain

 % Calculate the new state
 x = x_prediction + K * Y; % y
 P = (eye(5) - K * H) * P_prediction;
 estimated_positions(i, :) = x(1:2); 

end

N = estimated_positions;
x_coords = (N(:,1));
y_coords = (N(:,2));

output_data = [timestamps, estimated_positions];
writematrix(output_data, 'Recording_7_estimated_positions.csv');

% O = readtable("Data/Recording 6 true positions.csv");  
% x_real = table2array(O(:,2));
% y_real = table2array(O(:,3)); 

% 
O = readtable("Data/Recording 7.csv");  % comment out when not using Recording 7
x_real = table2array(O(1:1095,17));
y_real = table2array(O(1:1095,18));

plot(x_real,y_real,'red')
hold on;
plot(x_coords,y_coords,'blue')
hold off

%calculates the mean square error
mse_x = immse(x_real,x_coords)
mse_y = immse(y_real,y_coords)