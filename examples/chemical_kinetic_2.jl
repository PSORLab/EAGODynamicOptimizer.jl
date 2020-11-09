using JuMP, EAGODynamicOptimizer, DynamicBoundsBase,
      DynamicBoundspODEsIneq, DynamicBoundspODEsDiscrete

using DataFrames, CSV

data = CSV.read("kinetic_intensity_data.csv")

# Defines pODEs problem
x0(p) = [140.0; 0.4; 0.0; 0.0; 0.0]
function f!(dx, x, p, t)

    T = 273.0
    K2 = 46.0*exp(6500.0/T - 18.0)
    K3 = 2.0*K2
    k1 = 53.0
    k1s = k1*10^(-6)
    k5 = 0.0012
    cO2 = 0.002

    dx[1] = k1*x[4]*x[5]-cO2*(p[1]+p[2])*x[1] + p[1]*x[3]/K2+p[2]*x[2]/K3-k5*x[1]^2
    dx[2] = p[2]*cO2*x[1] - (p[2]/K3 + p[3])*x[2]
    dx[3] = p[1]*cO2*x[1] - p[1]*x[3]/K2
    dx[4] = -k1s*x[4]*x[5]
    dx[5] = -k1*x[4]*x[5]
    nothing
end
tspan = (0.0, 1.0)
pL = [10.0  10.0  0.001]
pU = [1200.0  1200.0  40.0]
pode_problem = ODERelaxProb(f!, tspan, x0, pL, pU)
set!(pode_problem, SupportSet([i for i in 0.0:0.01:2.0]))

# Initializes the Dynamic Extension
dynamic_ext = DynamicExt(DifferentialInequality(pode_problem,
                                                calculate_relax = true,
                                                calculate_subgradient = true))

m, y = EAGODynamicModel(dynamic_ext, "verbosity" => 1, "output_iterations" => 1)

# Defines function for intensity
intensity(xA,xB,xD) = xA + (2/21)*xB + (2/21)*xD

# Defines the objective: integrates the ODEs and calculates SSE
function objective(x, p)

    SSE = zero(typeof(p))
    for i = 1:200
        SSE += (intensity(x[1, t], x[2, t], x[3, t]) - data[t, :intensity][i])^2
    end
    return SSE
end

add_supported_objective!(m, objective)

optimize!(m)
objective_value = objective_value(m)
status = primal_status(m)
solution = value.(y)
