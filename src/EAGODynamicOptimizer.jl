module EAGODynamicOptimizer

using MathOptInterface, DocStringExtensions, EAGO, DynamicBoundsBase, JuMP
import EAGO: preprocess!, postprocess!, lower_problem!,
             upper_problem!, cut_condition, ExtensionType

import Base: getindex, setindex!

const MOI = MathOptInterface
const DBB = DynamicBoundsBase

export DynamicExt, EAGODynamicModel, supported_objective!

include("subproblems.jl")
include("model.jl")
include("state_vector.jl")
include("subproblems.jl")

end # module
