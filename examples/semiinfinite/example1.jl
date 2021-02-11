
#=
Make sure DynamicBounds.jl appropriately reexports functions
=#
using JuMP, EAGODynamicOptimizer, DynamicBounds

# Define semi-infinite program
f(x,u) = (1/3)*x[1]^2 + x[2]^2 + x[1]/2
gSIP(x,u,p) = (1.0 - (x[1]^2)*(p[1]^2))^2 - x[1]*p[1]^2 - x[2]^2 + x[2]

x_l = Float64[-1000.0, -1000.0]
x_u = Float64[1000.0, 1000.0]
p_l = Float64[0.0]
p_u = Float64[1.0]

#integrator =
#sip_dyn_ext = SIPDynamicExt(integrator)
sip_result = sip_solve(sip_dyn_ext, SIPResRev(), x_l, x_u, p_l, p_u,
                       f, Any[gSIP], abs_tolerance = 1E-3)

#=
# Defines pODEs problem
x0(p) = [1.2; 1.1]
function f!(dx, x, u, p, t)
    dx[1] = u[1]*x[1]*(one(typeof(p[1])) - x[2]) + p[1]
    dx[2] = u[1]*x[2]*(x[1] - one(typeof(u[1])))
    nothing
end
tspan = (0.0, 1.0)
xL = [2.95]
xU = [3.05]
uL = [2.95]
uU = [3.05]
pL = [0.001]
pU = [1.002]
pode_problem = ODERelaxProb(f!, tspan, x0, pL, pU)
set!(pode_problem, SupportSet([i for i in 0.0:0.01:1.0]))

m, y = SIPDynamicExt{T}()

# Adds objective function
function obj(x, u, p)
    x[1, 0.01]*p[1] + x[2, 0.05]
end
add_supported_objective!(m, () -> obj)

# Adds objective function
function gsip(x, u, p)
    x[1, 0.01]*p[1] + x[2, 0.05]
end
add_supported_constraint!(m, gsip)
=#
