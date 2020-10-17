module EAGODynamicOptimizer

using MathOptInterface, DocStringExtensions, EAGO, DynamicBounds
import EAGO: preprocess!, postprocess!, lower_problem!,
             upper_problem!, cut_condition, ExtensionType

import Base: getindex, setindex!

const MOI = MathOptInterface

include("subproblems.jl")

end # module
