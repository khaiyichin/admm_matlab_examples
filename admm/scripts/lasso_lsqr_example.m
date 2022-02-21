% L1-regularized least-squares example for sparse A

%% Generate problem data

randn('seed', 0);
rand('seed',0);

m  = 1000000;   % number of examples
n  = 10000;     % number of features
p1 = 0.001;     % sparsity density of solution vector 
p2 = 0.0001;    % sparsity density of A

x0 = sprandn(n, 1, p1);
A = sprandn(m, n, p2);
b = A*x0 + 0.1*randn(m,1);

lambda_max = norm(A'*b, 'inf');
lambda = 0.1*lambda_max; 

%% Solve problem

[x history] = lasso_lsqr(A, b, lambda, 1.0, 1.0);

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
