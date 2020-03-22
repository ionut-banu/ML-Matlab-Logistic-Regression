%% Logistic Regression 
%

%% Initialization
clear ; close all; clc

%% Part 1 - Load Data

fid = fopen('crx.data');
FC = textscan(fid, '%s%f%f%s%s%s%s%s%s%s%s%s%s%s%s%s', 'Delimiter', ',');
fclose(fid);

Xtrain = [FC{1,2} FC{1,3}]; 
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

[m, n] = size(Xtrain);

Xtrain = [ones(mtrain, 1), Xtrain]; % add a column of ones to x
initial_theta = zeros(n+1, 1); % initialize fitting parameters

fprintf('\nComputing initial cost ...\n')
[cost, grad] = costFunction(initial_theta, Xtrain, ytrain);
fprintf('Cost at initial theta: %f\n', cost);
fprintf('Gradient at initial theta: \n');
fprintf(' %f \n', grad);

fprintf('Press Enter to continue\n')
pause;

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, Xtrain, ytrain)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, Xtrain, ytrain);

hold on;
% Labels and Legend
xlabel('Var 1')
ylabel('Var 2')

legend('Approved', 'Not approved', 'Decision Boundry');
hold off;












