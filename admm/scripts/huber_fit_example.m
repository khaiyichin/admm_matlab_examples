% Huber function fitting example

%% Generate problem data
randn('seed', 0);
rand('seed',0);

m = 5000;       % number of examples
n = 200;        % number of features

x0 = randn(n,1);
A = randn(m,n);
A = A*spdiags(1./norms(A)',0,n,n); % normalize columns
b = A*x0 + sqrt(0.01)*randn(m,1);
b = b + 10*sprand(m,1,200/m);      % add sparse, large noise

%% Solve problem

[x history] = huber_fit(A, b, 1.0, 1.0);

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
