using JuMP, EAGODynamicOptimizer, DynamicBoundsBase,
      DynamicBoundspODEsIneq, DynamicBoundspODEsDiscrete

using DataFrames, CSV

data = CSV.read("C:\\Users\\wilhe\\Desktop\\Package Development\\EAGODynamicOptimizer.jl\\examples\\kinetic_intensity_data.csv", DataFrame)
data_dict = Dict{Float64,Float64}()
for r in eachrow(data)
    data_dict[r.time] = r.intensity
end

# Defines pODEs problem
x0(p) = [0.0; 0.0; 0.0; 0.4; 140.0]
function RHS!(dx, x, p, t)

    T = 273.0
    K2 = 46.0*exp(6500.0/T - 18.0)
    K3 = 2.0*K2
    k1 = 53.0
    k1s = k1*10^(-6)
    k5 = 0.0012
    cO2 = 0.002

    dx[1] = k1*x[4]*x[5] -cO2*(p[1]+p[2])*x[1] + p[1]*x[3]/K2+p[2]*x[2]/K3-k5*x[1]*x[1]
    dx[2] = p[2]*cO2*x[1] - (p[2]/K3 + p[3])*x[2]
    dx[3] = p[1]*cO2*x[1] - p[1]*x[3]/K2
    dx[4] = -k1s*x[4]*x[5]
    dx[5] = -k1*x[4]*x[5]
    nothing
end
tspan = (0.0, 2.0)
pL = [10.0;  10.0;  0.001]
pU = [1200.0;  1200.0;  40.0]
pode_problem = ODERelaxProb(RHS!, tspan, x0, pL, pU)

set!(pode_problem, SupportSet([i for i in 0.01:0.01:2.0]))

# define constant state bounds
xL = zeros(5)
xU = [140.0; 140; 140.0; 0.4; 140.0]
set!(pode_problem, ConstantStateBounds(xL, xU))

# define polyhedral bounds
A = []
b = []
#PolyhedralConstraint(A, b)

ticks = 100.0
steps = 200.0
tend = 1*steps/ticks # lo 7.6100
tol = 1E-5

# Initializes the Dynamic Extension
#=
dynamic_ext = DynamicExt(DifferentialInequality(pode_problem,
                                                calculate_relax = true,
                                                calculate_subgradient = true))
=#
dynamic_ext = DynamicExt(DiscretizeRelax(pode_problem, DynamicBoundspODEsDiscrete.LohnerContractor{8}(),
                                         h = 1/ticks, repeat_limit = 1, skip_step2 = false,
                                         step_limit = steps, relax = false, tol= tol))

m, y = EAGODynamicModel(dynamic_ext, "verbosity" => 1,
                        "output_iterations" => 100,
                        "time_limit" => 7200.0,
                        "log_on" => true,
                        "log_interval" => 1000)

# Defines function for intensity
intensity(xA,xB,xD) = xA + (2/21)*xB + (2/21)*xD

# Defines the objective: integrates the ODEs and calculates SSE
function objective_data(x, p, data_dict)
    SSE = zero(typeof(p[1]))
    for t = 0.01:0.01:2.0
        val = data_dict[t]
        SSE += (intensity(x[1, t], x[2, t], x[3, t]) - val)^2
    end
    return SSE
end
objective(x, p) = objective_data(x, p, data_dict)

add_supported_objective!(m, objective)

println(" ")
println("optimize start")
optimize!(m)
bm = backend(m)
bm_ext = bm.optimizer.model.optimizer.ext_type
println("optimize end")

obj_value = objective_value(m)
status = primal_status(m)
solution = value.(y)
