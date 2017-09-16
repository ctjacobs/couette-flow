% Solves the equation d2u/dy2 = 0 to simulate Couette flow.
% This considers the fluid between two parallel plates located at y = 0 and y = Ly, with the bottom
% plate stationary and the top and the top plate moving at a constant
% speed U. A shear stress du/dy develops in the fluid as a result.
%
% Note that the time-dependent PDE du/dt = nu*(d2u/dy2) is actually solved
% here, where nu is the kinematic viscosity of the fluid. It is the final
% steady-state solution which satisfies d2u/dy2 = 0.
%
% Copyright (C) 2017 Christian Thomas Jacobs

function u = couette(Ly, Ny, T, dt, U, nu)
    % Grid spacing in the y direction.
    dy = Ly/(Ny-1);

    % Number of timesteps.
    Nt = ceil(T/dt);
    
    % The stored solution from the previous timestep.
    % Initially this is a zero velocity field.
    u_old = zeros(Ny, 1);
    
    % Timestepping loop.
    % This uses the Crank-Nicolson timestepping scheme.
    for n = 1:Nt
        % Compute solution at timestep n.
        u = solve(u_old, U, dy, dt, nu)
        
        % Save the solution for use in the next timestep.
        u_old = u;
    end

    % Compute the exact solution.
    u_exact = exact(U, Ly, Ny);
    
    % Determine maximum error (in the Euclidean norm) of the numerical
    % solution.
    error = sum(abs(u-u_exact).^2)^0.5;
    fprintf('The error in the Euclidean norm is %.5f\n', error);
    
    % Plot the numerical and exact solutions.
    figure(1)
    plot(linspace(0, Ly, Ny), u)
    legend('Numerical')
    xlabel('y')
    ylabel('u(y)')
    
    % Plot the error.
    figure(2)
    plot(linspace(0, Ly, Ny), abs(u-u_exact))
    legend('Error')
    xlabel('y')
    ylabel('|u(y) - u_{exact}(y)|')    
end

function u = solve(u_old, U, dy, dt, nu)
    % Step forward one timestep by solving the discretised system of
    % equations.
    
    % Setup solution vector.
    Ny = size(u_old, 1);
    u = zeros(Ny, 1);

    % Thomas algorithm for the solution of the tri-diagonal matrix system.
    c1 = -nu*dt/(2*dy^2);
    c2 = 1 + nu*dt/(dy^2);
    a = zeros(Ny-1, 1);
    b = zeros(Ny, 1);
    r = zeros(Ny, 1);
 
    % Forward sweep. Note that indices 1 and Ny are not considered here, since
    % u(1) and u(Ny) are known from the boundary conditions.
    r(2) = (-c2+2)*u_old(2) - c1*u_old(3);  % Apply Dirichlet condition u(1) = 0.
    a(2) = c1/c2;
    b(2) = r(2)/c2;
    for j = 3:Ny-2
        r(j) = (-c2+2)*u_old(j) - c1*u_old(j+1) - c1*u_old(j-1);
        a(j) = c1/(c2-c1*a(j-1));
        b(j) = (r(j) - c1*b(j-1))/(c2 - c1*a(j-1));
    end
    r(Ny-1) = (-c2+2)*u_old(Ny-1) - 2*c1*U - c1*u_old(Ny-2);  % Apply Dirichlet condition u(Ny) = U.
    b(Ny-1) = (r(Ny-1) - c1*b(Ny-2))/(c2 - c1*a(Ny-2));
    
    % Back substitution.
    u(Ny-1) = b(Ny-1);
    for j = Ny-2:-1:2
        u(j) = b(j) - a(j)*u(j+1);
    end
    
    % Enforce boundary conditions.
    u(1) = 0;
    u(Ny) = U;

end

function u = exact(U, Ly, Ny)
    % Exact solution given by u(y) = U*(y/Ly).
    u = zeros(Ny, 1);
    for j = 1:Ny
        y = (j-1)*(Ly/(Ny-1));
        u(j) = U*(y/Ly);
    end
end
