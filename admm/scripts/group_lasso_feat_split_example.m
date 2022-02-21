% Group lasso example with random data

%% Generate problem data

randn('seed', 0);
rand('seed',0);

m = 200;        % amount of data
K = 200;        % number of blocks
ni = 100;       % size of each block

n = ni*K;
p = 10/K;      % sparsity density  

% generate block sparse solution vector
x = zeros(ni,K);
for i = 1:K,
    if( rand() < p)
        % fill nonzeros
        x(:,i) = randn(ni,1);
    end
end
x = vec(x);

% generate random data matrix
A = randn(m,n);

% normalize columns of A
A = A*spdiags(1./norms(A)',0,n,n);

% generate measurement b with noise
b = A*x + sqrt(1)*randn(m,1);

% lambda max
for i = 1:K,
    Ai = A(:,(i-1)*ni + 1:i*ni);
    nrmAitb(i) = norm(Ai'*b);
end
lambda_max = max( nrmAitb );

% regularization parameter
lambda = 0.5*lambda_max;

xtrue = x;   % save solution

%% Solve problem
[x history] = group_lasso_feat_split(A, b, lambda, ni, 10, 1.0);

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

