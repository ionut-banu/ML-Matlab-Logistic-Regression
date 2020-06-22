%% Logistic Regression Regularized
%

%% Initialization
clear ; close all; clc

%% Part 1 - Load Data

fid = fopen('crx.data');
FC = textscan(fid, '%s%f%f%s%s%s%s%f%s%s%s%s%s%s%s%s', 'Delimiter', ',');
fclose(fid);

Xtrain = [FC{1,2} FC{1,8}]; 
ytrain = cell2mat(FC{1,16}) == '+';
mtrain = length(ytrain); % number of training examples

plotLogisticData(Xtrain, ytrain);

hold on;
% Labels and Legend
xlabel('Var 1')
ylabel('Var 2')

legend('Approved', 'Not approved')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Part 2 - Run gradient descent

% Add Polynomial Features
Xtrain = mapFeature(Xtrain(:,1), Xtrain(:,2));

initial_theta = zeros(size(Xtrain, 2), 1); % initialize fitting parameters

lambda = 2; % regularization parameter lambda to 1

fprintf('\nComputing initial cost ...\n')
[cost, grad] = costFunctionReg(initial_theta, Xtrain, ytrain, lambda);
fprintf('Cost at initial theta: %f\n', cost);
fprintf('Gradient at initial theta, display first 10 values only: \n');
fprintf(' %f \n', grad(1:10));

fprintf('Press Enter to continue\n')
pause;

initial_theta = zeros(size(Xtrain, 2), 1);

lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, Xtrain, ytrain, lambda)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta, display first 10 values only: \n');
fprintf(' %f \n', theta(1:10));

% Plot Boundary
plotDecisionBoundary(theta, Xtrain, ytrain);

hold on;
% Labels and Legend
xlabel('Var 1')
ylabel('Var 2')

legend('Approved', 'Not approved', 'Decision Boundry');
hold off;
