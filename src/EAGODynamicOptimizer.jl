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
      DynamicBoundsBase, JuMP, Reexport, Ipopt,
      ModelingToolkit, OrdinaryDiffEq

using SourceCodeMcCormick

using ForwardDiff: Dual, Partials, construct_seeds, partials

using DynamicBoundspODEsIneq

@reexport using EAGO

import EAGO: _ext, _relaxed_optimizer, _variable_num, branch_node!, 
             build_model, convergence_check!, cut_condition, DefaultExt, 
             ExtensionType, fathom!, FullVar, GlobalOptimizer, initial_parse!, 
             is_integer, log_iteration!, lower_bound, lower_problem!, 
             map_argmax, MINCVX, node_selection!, NodeBB, optimize_hook!, 
             parse_global!, postprocess!, preprocess!, presolve_global!, 
             presolve_global!, print_interation!, print_node!, print_preamble!, 
             print_results!, select_branch_variable, set_global_lower_bound!, 
             set_tolerance!, sip_bnd!, sip_llp!, unpack_global_solution!, 
             update_relaxed_problem_box!, upper_bound, upper_problem!

             
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
