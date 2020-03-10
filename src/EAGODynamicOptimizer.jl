module EAGODynamicOptimizer

using MathOptInterface, DocStringExtensions, EAGO, DynamicBounds, McCormick
import EAGO: relax_problem!, upper_problem!, preprocess!, postprocess!, cut_condition
import Base: getindex

const MOI = MathOptInterface

export StateCalc, SupportedScalarFunction, SupportedVectorFunction, relax, bound
include("functions.jl")

end # module
