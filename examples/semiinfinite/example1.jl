
#=
Make sure DynamicBounds.jl appropriately reexports functions
=#
using JuMP, EAGODynamicOptimizer, DynamicBounds

# Defines pODEs problem
x0(p) = [1.2; 1.1]
function f!(dx, x, u, p, t)
    dx[1] = u[1]*x[1]*(one(typeof(p[1])) - x[2]) + p[1]
    dx[2] = u[1]*x[2]*(x[1] - one(typeof(u[1])))
    nothing
end
tspan = (0.0, 1.0)
uL = [2.95]
uU = [3.05]
uL = [0.001]
uU = [1.002]
pode_problem = ODERelaxProb(f!, tspan, x0, pL, pU)
set!(pode_problem, SupportSet([i for i in 0.0:0.01:1.0]))
