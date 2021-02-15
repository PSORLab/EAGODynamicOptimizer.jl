# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# EAGODynamicOptimizer.jl
# See https://github.com/PSORLab/EAGODynamicOptimizer.jl
#############################################################################
# src/EAGODynamicOptimizer.jl
# Main module loading for EAGODynamicOptimizer.
#############################################################################

module EAGODynamicOptimizer

using MathOptInterface, DocStringExtensions,
      DynamicBoundsBase, JuMP, Reexport

@reexport using EAGO

import EAGO: preprocess!, postprocess!, lower_problem!,
             upper_problem!, cut_condition, ExtensionType

import Base: getindex, setindex!

const MOI = MathOptInterface
const DBB = DynamicBoundsBase

export DynamicExt, EAGODynamicModel, SIPDynamicExt,
       add_supported_objective!, add_supported_constraint!

include("global/state_vector.jl")
include("global/global_extension.jl")
include("global/subroutines.jl")
include("global/model.jl")
include("semiinfinite/sip_extension.jl")


end # module
