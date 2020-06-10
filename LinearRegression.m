%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc
format short g;
fprintf('Loading data ...\n');

%% Load Data
%nombrearchivo = input('Introduce the name of the data file:');
data = load('datos.txt');
[trainingexamples numberfeaturesy] = size(data);
X = data(:, 1:(numberfeaturesy-1));
y = data(:, numberfeaturesy);
m = length(y);

%Showing the data
figure (1);
scatter3(X(:,1),X(:,2),y,'filled');


% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

figure (2);
scatter3(X(:,1),X(:,2),y,'filled');

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(numberfeaturesy, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

%Plot H_theta(X)

tx = linspace (-3, 3, 100);
ty = linspace (-3, 3, 100);
[xx, yy] = meshgrid (tx, ty);
tz = theta(1)+theta(2)*xx+theta(3)*yy;
figure(3);
mesh (tx, ty, tz);
xlabel('tx');
ylabel('ty');
zlabel('tz');
title('3-D plot')

% Plot the convergence graph
figure(4);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%% Estimate the price of a 1650 sq-ft, 3 br house
%% ====================== YOUR CODE HERE ======================
%% Recall that the first column of X is all-ones. Thus, it does
%% not need to be normalized.
X_example = input('Intruce your features: (in a row vector form)');
X_example = (X_example-mu)./sigma;
X_example = [1 X_example];
price = theta'*X_example'; % You should change this
%
%
%% ============================================================
%
fprintf(['Predicted price is ' ...
         '(using gradient descent):\n $%f\n'], price);
%
%fprintf('Program paused. Press enter to continue.\n');
%pause;
