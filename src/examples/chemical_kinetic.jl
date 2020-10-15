using JuMP, EAGODynamicOptimizer, DynamicBounds

x0(p) = [1.2; 1.1]
function f!(dx, x, p, t)
    dx[1] = p[1]*x[1]*(one(typeof(p[1])) - x[2])
    dx[2] = p[1]*x[2]*(x[1] - one(typeof(p[1])))
    nothing
end
tspan = (0.0, tend)
pL = [2.95]
pU = [3.05]

dynamic_ext = DynamicExt()
pode_problem = ODERelaxProb(f!, tspan, x0, pL, pU)
set_integrator!(dynamic_ext, DiscretizeRelax(pode_problem))

add_support_set!(dynamic_ext, SupportSet([i for i in 0.0:0.01:2.0]))

function obj(x, p)
    x(1, 0.01)*p[1] + x(2, 0.05)
end
add_supported_objective!(dynamic_ext, obj)

m = EAGODynamicModel()
@variable(m, pL[i] <= p[i = 1:5] <= pU[i])
obj_value = objective_value(m)
