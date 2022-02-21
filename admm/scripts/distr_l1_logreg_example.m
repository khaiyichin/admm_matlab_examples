% Distributed L1 regularized logistic regression
% (compared against l1_logreg package)

%% Generate problem data
rand('seed', 0);
randn('seed', 0);

n = 200; 
m = 200; 
N = 100;

w = sprandn(n, 1, 100/n);       % N(0,1), 10% sparse
v = randn(1);                  % random intercept

X0 = sprandn(m*N, n, 10/n);           % data / observations
btrue = sign(X0*w + v);

% noise is function of problem size use 0.1 for large problem
b0 = sign(X0*w + v + sqrt(0.1)*randn(m*N, 1)); % labels with noise

% packs all observations in to an m*N x n matrix
A0 = spdiags(b0, 0, m*N, m*N) * X0;

ratio = sum(b0 == 1)/(m*N);
mu = 0.1*1/(m*N) * norm((1-ratio)*sum(A0(b0==1,:),1) + ratio*sum(A0(b0==-1,:),1), 'inf');

x_true = [v; w];

%% Solve problem

[x history] = distr_l1_logreg(A0, b0, mu, N, 1.0, 1.0);

%% Reporting

K = length(history.objval);                                                                                                        

h = figure;
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2); 
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

g = figure;
subplot(2,1,1);                                                                                                                    
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2); 
ylabel('||r||_2'); 

subplot(2,1,2);                                                                                                                    
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);   
ylabel('||s||_2'); xlabel('iter (k)'); 
