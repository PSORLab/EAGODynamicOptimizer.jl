using JuMP, EAGODynamicOptimizer, DynamicBoundsBase,
      DynamicBoundspODEsIneq, DynamicBoundspODEsDiscrete

# Defines pODEs problem
x0(p) = [1.2; 1.1]
function f!(dx, x, p, t)
    dx[1] = p[1]*x[1]*(one(typeof(p[1])) - x[2])
    dx[2] = p[1]*x[2]*(x[1] - one(typeof(p[1])))
    nothing
end
tspan = (0.0, 1.0)
pL = [2.95]
pU = [3.05]
pode_problem = ODERelaxProb(f!, tspan, x0, pL, pU)
set!(pode_problem, SupportSet([i for i in 0.0:0.01:2.0]))

# Initializes the Dynamic Extension
#=
dynamic_ext = DynamicExt(DifferentialInequality(pode_problem,
                                                calculate_relax = false,
                                                calculate_subgradient = false))
=#

steps = 100
dynamic_ext = DynamicExt(DiscretizeRelax(pode_problem,
                                         DynamicBoundspODEsDiscrete.LohnerContractor{7}(),
                                         repeat_limit = 1, skip_step2 = false,
                                         step_limit = steps, relax = false))

# Creates Model with dynamic extension
m, y = EAGODynamicModel(dynamic_ext, "verbosity" => 4)

# Adds objective function
function obj(x, p)
    x(1, 0.01)*p[1] + x(2, 0.05)
end
add_supported_objective!(m, obj)

optimize!(m)

obj_value = objective_value(m)
