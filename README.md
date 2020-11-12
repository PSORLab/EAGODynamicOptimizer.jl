# EAGODynamicOptimizer.jl
Extension of the EAGO to Problems with Parametric Differential Equation Constraints

## Intended Scope

EAGODynamicOptimizer.jl is an extension to the EAGO.jl solver which makes use
of approaches for computing reachability bounds of differential equations to
solve optimization problems with an embedded differential equations to a certificate of global optimality.

## Current State

- Currently, only supports box-constrained problems with a `ODERelaxProb` embedded.
- Future functionality is intended to support constrained problems.
- Additional, `AbstractDERelaxProblems` will be supported in the future as novel
methods become available to solve them.
- Once a sufficient body of methods is provided. We'll look into supporting `InfiniteOpt.jl`
as front end for this.

## Key functions

- `DynamicExt(integrator)`: creates a `DynamicExt::EAGO.ExtensionType` structure
that holds the buffer used by EAGO to compute relaxations and bounds of the
state variables and in turn supported objectives and constraints for a given
`integrator::AbstractDERelaxIntegrator` (currently only `ODERelaxProb` is supported).
- `EAGODynamicModel(ext::DynamicExt, kwargs...)`:
- `add_supported_objective!(t::Model, obj)`:

## Usage

In the below section, EAGODynamicOptimizer.jl is used solve the kinetic
parameter estimation problem [1,2] wherein the problem is integrated using
DifferentialInequality [3,4,5] method to construct state relaxations.

First, we load the required modules and pull in the dataset needed by the
objective function.
```julia
using JuMP, EAGODynamicOptimizer, DynamicBounds, DataFrames, CSV

data = CSV.read("kinetic_intensity_data.csv", DataFrame)
data_dict = Dict{Float64,Float64}()
for r in eachrow(data)
    data_dict[r.time] = r.intensity
end
```

We now define the parametric differential equation system we wish to embedded
in the optimizer.
```julia
x0(p) = [0.0; 0.0; 0.0; 0.4; 140.0]
function rhs!(dx, x, p, t)

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
pode_problem = ODERelaxProb(rhs!, tspan, x0, pL, pU)
```

We then append any other important attributes to the problem of interest. If
we wanted to specify box constraints on the state variables we could do so as follows:
```julia
# define constant state bounds
xL = zeros(5)
xU = [140.0; 140; 140.0; 0.4; 140.0]
set!(pode_problem, ConstantStateBounds(xL, xU))
```

Next, we create a `DynamicExt` that uses our preferred integrator. In this case,
will use the `DifferentialInequality` integrator. EAGO is then initialized
and a JuMP model is returned with the EAGO optimizer with the specified
`DynamicExt` set. Other options for the EAGO solver may be set via keyword arguments
as usual.
```julia
dynamic_ext = DynamicExt(DifferentialInequality(pode_problem,
                                                calculate_relax = true,
                                                calculate_subgradient = true))
m, y = EAGODynamicModel(dynamic_ext, "verbosity" => 1, "output_iterations" => 1)
```

An objective function is then defined. For the `ODERelaxProb`, state variables
may be referenced using the syntax `x[i,t]` where `i` is the ith component of the
state vector and `t` is the independent variable.
```julia
# Defines function for intensity
intensity(xA,xB,xD) = xA + (2/21)*xB + (2/21)*xD

# Adds objective function
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
```

Lastly, we retrieve key information about the solution of the optimization problem
```julia
obj_value = objective_value(m)
status = primal_status(m)
solution = value.(y)
```

## References
1. Taylor, James W., et al. *Direct measurement of the fast, reversible addition of oxygen to cyclohexadienyl radicals in nonpolar solvents.* The Journal of Physical Chemistry A 108.35 (2004): 7193-7203.
2. Singer, Adam B., and Paul I. Barton. *Global optimization with nonlinear ordinary differential equations.* Journal of Global Optimization 34.2 (2006): 159-190.
3. JK Scott, PI Barton, *Bounds on the reachable sets of nonlinear control systems*,
  Automatica 49 (1), 93-100
4. JK Scott, PI Barton, *Improved relaxations for the parametric solutions of ODEs using differential inequalities*, Journal of Global Optimization, 1-34
5. JK Scott, *Reachability Analysis and Deterministic Global Optimization of Differential-Algebraic Systems*, Massachusetts Institute of Technology
