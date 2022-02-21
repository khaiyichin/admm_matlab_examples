function [z, history] = polyhedra_intersection(A1, b1, A2, b2, rho, alpha)
% polyhedra_intersection  Find point in intersection of two polyhedra via 
%  ADMM
%
% [x, history] = polyhedra_intersection(A1, b1, A2, b2, rho, alpha)
% 
% Solves the following problem via ADMM:
% 
%   find         x
%   subject to   x in { y | A1 y <= b1 }
%                x in { y | A2 y <= b2 }.
% 
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and 
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
% 
% rho is the augmented Lagrangian parameter. 
%
% alpha is the over-relaxation parameter (typical values for alpha are 
% between 1.0 and 1.8).
% 
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

n = size(A1,2);

%% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % x-update
    % use cvx to find point in first polyhedra
    cvx_begin quiet
        variable x(n)
        minimize (sum_square(x - (z - u)))
        subject to
            A1*x <= b1
    cvx_end

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    % use cvx to find point in second polyhedra
    cvx_begin quiet
        variable z(n)
        minimize (sum_square(x_hat - (z - u)))
        subject to
            A2*z <= b2
    cvx_end

    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks

    history.objval(k)  = 0;
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

if ~QUIET
    toc(t_start);
end

end
