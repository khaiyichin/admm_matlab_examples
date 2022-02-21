% Cardinality constrained least-squares example (nonconvex)

%% Generate problem data

randn('seed', 0);
rand('seed',0);

m = 1500;       % number of examples
n = 5000;       % number of features
p = 100/n;      % sparsity density  

% generate sparse solution vector
x = sprandn(n,1,p);

% generate random data matrix
A = randn(m,n);

% normalize columns of A
A = A*spdiags(1./sqrt(sum(A.^2))', 0, n, n);

% generate measurement b with noise
b = A*x + sqrt(0.001)*randn(m,1);

xtrue = x;   % save solution

%% Solve problem

[x history] = regressor_sel(A, b, p*n, 1.0);

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

%% Compare to l1 regularization

% err1 = [];
% card1 = [];
% for K = 1:5:2*p*n
%     [x history] = regressor_sel(A, b, K, rho);
%     err1 = [err1 norm(A*x - b)];
%     card1 = [card1 K];
% end
% 
% % lambda max
% lambda_max = norm( A'*b, 'inf' );
% 
% err2 = [];
% card2 = [];
% for lambda = (1:-.01:.02)*lambda_max
%     [x history] = lasso(A, b, lambda, rho, 1);
%     err2 = [err2 norm(A*x - b)];
%     card2 = [card2 sum(x~=0)];
% end
% 
% p = figure
% stairs(card1, err1, 'k', 'LineWidth', 2); 
% hold on;
% stairs(card2, err2, 'k--', 'LineWidth', 2);
% ylabel('||Ax-b||'); xlabel('card(x)');
% legend('regressor selection', 'l1 regularization');
