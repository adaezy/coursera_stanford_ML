
% Linear regression with one variable
%Plotting the data
data = load('ex1data1.txt'); % read comma separated data
X = data(:, 1); y = data(:, 2);
plotData(X,y) 

%Gradient Descent
% Update Equations
%Implementation
m = length(X); % number of training examples
X = [ones(m,1),data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

% Computing the cost 
% Compute and display initial cost with theta all zeros
computeCost(X, y, theta)

% Compute and display initial cost with non-zero theta
computeCost(X, y,[-1; 2])

% Gradient descent
% Run gradient descent:
% Compute theta
theta = gradientDescent(X, y, theta, alpha, iterations);

% Print theta to screen
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure


% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);
 

%Linear regression with multiple variables
% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
% First 10 examples from the dataset

fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

%Feature Normalization
% Scale features and set them to zero mean

[X, mu, sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X];

%Gradient Descent

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
given_value = ([1650,3].* mu) ./ sigma;
price = [1,[given_value]]* theta; % Enter your price formula here


% ============================================================

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);

%MATLAB Tip: To compare how different learning learning rates affect convergence, it's helpful to plot J for several learning rates on the same figure. In MATLAB, this can be done by performing gradient descent multiple times with a hold on command between plots. Make sure to use the hold off command when you are done plotting in that figure. Concretely, if you've tried three different values of alpha (you should probably try more values than this) and stored the costs in J1, J2 and J3, you can use the following commands to plot them on the same figure:
%     plot(1:50, J1(1:50), 'b');
%     hold on
%     plot(1:50, J2(1:50), 'r');
%     plot(1:50, J3(1:50), 'k');
%     hold off

% Run gradient descent:
% Choose some alpha value
%alpha = 0.01;
%alpha = 0.2
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[~, J1] = gradientDescentMulti(X, y, theta, 0.1, num_iters);
[~, J2] = gradientDescentMulti(X, y, theta, 0.08, num_iters);
[~, J3] = gradientDescentMulti(X, y, theta, 0.05, num_iters);

% Plot the convergence graph
plot(1:num_iters, J1, '-b', 'LineWidth', 2);
hold on
plot(1:num_iters, J2, 'r', 'LineWidth', 2);
plot(1:num_iters, J3, 'k', 'LineWidth', 2);
hold off
xlabel('Number of iterations');
ylabel('Cost J');
clf('reset')

% Run gradient descent
% Replace the value of alpha below best alpha value you found above
alpha = 0.08;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))

% Estimate the price of a 1650 sq-ft, 3 br house. You can use the same
% code you entered ealier to predict the price
% ====================== YOUR CODE HERE ======================
given_value = ([1650,3].* mu) ./ sigma;
price = [1,[given_value]]* theta; % Enter your price formula here
% ============================================================
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);

% Normal Equation
% Solve with normal equations:
% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations:\n%f\n%f\n%f', theta(1),theta(2),theta(3));

% Estimate the price of a 1650 sq-ft, 3 br house. 
% ====================== YOUR CODE HERE ======================
price = [1,1650,3]* theta; 
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f', price);     
