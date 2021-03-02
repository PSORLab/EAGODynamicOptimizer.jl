
#=
Make sure DynamicBounds.jl appropriately reexports functions
=#
using JuMP, EAGODynamicOptimizer, DynamicBoundsBase, DynamicBoundspODEsIneq

# Define semi-infinite program
obj(u) = (1/3 + 1)*u[1]^2 + u[1]/2
gSIP(y, u, p) = y[1,0.05]*u[1] + p[1] - 4.5

# Defines dynamics
y0(u, p) = [1.2; 1.1]
function f!(dy, y, u, p, t)
    dy[1] = p[1]*y[1]*(one(typeof(p[1])) - y[2])*u[1]
    dy[2] = p[1]*y[2]*(y[1] - one(typeof(p[1])))
    nothing
end
tspan = (0.0, 1.0)

u_l = Float64[1.1]
u_u = Float64[2.2]

pL = Float64[2.95]
pU = Float64[3.05]

# setup problem for dynamics
pode_problem = ODERelaxProb(f!, tspan, y0, pL, pU, nx = 2, param_num = length(pL))
set!(pode_problem, SupportSet([i for i in 0.0:0.01:1.0]))

# setup integrator factory
integrator_factory = prob -> DifferentialInequality(prob, calculate_relax = false, calculate_subgradient = false)

# create extension and solve problem
sip_dyn_ext = SIPDynamicExt(integrator_factory, pode_problem)
sip_result = sip_solve(SIPResRev(), u_l, u_u, pL, pU, obj, Any[gSIP], d = sip_dyn_ext, abs_tolerance = 1E-3)
