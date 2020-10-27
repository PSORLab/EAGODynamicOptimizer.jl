# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# EAGODynamicOptimizer.jl
# See https://github.com/PSORLab/EAGODynamicOptimizer.jl
#############################################################################
# src/model.jl
# Creates a JuMP model with a EAGO.Optimizer and a DynamicExt set.
#############################################################################

function EAGODynamicModel(ext::DynamicExt, kwargs...)

    # get bounds on decision variable
    pL = DBB.getall(ext.integrator, DBB.ParameterBound{Lower}())
    pU = DBB.getall(ext.integrator, DBB.ParameterBound{Upper}())
    np = DBB.get(ext.integrator, DBB.ParameterNumber())

    # initialize model and variables
    m = Model(optimizer_with_attributes(EAGO.Optimizer, kwargs...))
    set_optimizer_attribute(m, "ext_type", ext)
    set_optimizer_attribute(m, "branch_variable", Bool[true for i in 1:np])

    p = @variable(m, pL[i] <= p[i = 1:np] <= pU[i])

    return m, p
end
