using JuMP, EAGODynamicOptimizer, DynamicBoundsBase,
      DynamicBoundspODEsIneq, DynamicBoundspODEsDiscrete

# Defines pODEs problem
zones = 1

x0(p) = [1.0; 0.0]
function rhs!(dx, x, p, t, nz)

    pc = p[1]
    k1 = exp(8.86)*exp(-10215.4*pc)
    k2 = exp(24.25)*exp(-18820.5*pc)
    k3 = exp(23.67)*exp(-17008.9*pc)
    k4 = exp(18.75)*exp(-14190.8*pc)
    k5 = exp(20.70)*exp(-15599.8*pc)
    dx[1] = -k1*x[1] - (k3 + k4 + k5)*x[1]*x[2]
    dx[2] = k1*x[1] - k2*x[2] + k3*x[1]*x[2]
    nothing
end
rhs!(dx, x, p, t) = rhs!(dx, x, p, t, zones)

tol = 1E-5

tspan = (0.0, 11.0)
pL = ones(zones)*(698.15/748.15)
pU = ones(zones)
pode_problem = ODERelaxProb(rhs!, tspan, x0, pL, pU)
set!(pode_problem, SupportSet([i for i in 0.2:0.2:11.0]))

# define constant state bounds
#xL = zeros(1)
#xU = [140.0; 140; 140.0; 0.4; 140.0]
#set!(pode_problem, ConstantStateBounds(xL, xU))

dynamic_ext = DynamicExt(DifferentialInequality(pode_problem,
                                                calculate_relax = false,
                                                calculate_subgradient = false))

#=
dynamic_ext = DynamicExt(DiscretizeRelax(pode_problem, DynamicBoundspODEsDiscrete.LohnerContractor{10}(),
                                         repeat_limit = 1, skip_step2 = false,
                                         step_limit = 500, relax = false, tol= tol))

=#
m, y = EAGODynamicModel(dynamic_ext, "verbosity" => 4,
                        "output_iterations" => 1,
                        "time_limit" => 300.0,
                        "log_on" => true,
                        "log_interval" => 1000)

objective(x, p) = -x[2, 10.0]
add_supported_objective!(m, objective)

optimize!(m)
