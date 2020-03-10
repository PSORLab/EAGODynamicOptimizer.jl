"""
$(TYPEDEF)

Defines a parametric system of ODEs where `f` exist.
$(TYPEDFIELDS)
"""
struct DifferentialOptimizer <: MOI.AbstractOptimizer
    optimizer::EAGO.Optimizer
end

function DifferentialOptimizer(;options...) #TODO
end

function MOI.empty!(m::DifferentialOptimizer) #TODO
end

function MOI.is_empty(m::DifferentialOptimizer) #TODO
end

function MOI.copy_to(model::DifferentialOptimizer, src::MOI.ModelLike; copy_names = false)
    return MOI.Utilities.default_copy_to(model, src, copy_names)
end

function MOI.set(m::DifferentialOptimizer, ::MOI.Silent, value)
     MOI.set(m.optimizer, MOI.Silent(), value)
     return
end

for trait in (MOI.ObjectiveValue, MOI.NumberOfVariables, MOI.ObjectiveBound,
              MOI.RelativeGap, MOI.TerminationStatus, MOI.PrimalStatus,
              MOI.SolveTime, MOI.NodeCount)
    @eval function MOI.get(m::DifferentialOptimizer, ::$trait)
        MOI.get(m.optimizer, $trait)
    end
end

function MOI.get(m::DifferentialOptimizer, ::MOI.SolverName)
    return "EAGO Differential: Deterministic Global Dynamic Optimization"
end


function MOI.get(model::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    check_inbounds!(model, vi)
    return model._continuous_solution[vi.value]
end

function MOI.set(m::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    MOI.set(m.otpimizer, ::MOI.ObjectiveSense, sense)
    return
end

function MOI.set(m::Optimizer, ::MOI.ObjectiveFunction, func::SupportedFunction)
    check_inbounds!(m, func)
    m._objective_sv = func
    return
end

function MOI.add_constraint(m::Optimizer, v::SupportedFunction, lt::LT)
    # Adds storage...
    return CI{SupportedFunction, LT}(vi.value)
end

function MOI.add_constraint(m::Optimizer, v::SupportedFunction, gt::GT)
    return CI{SupportedFunction, GT}(vi.value)
end

function MOI.add_constraint(m::Optimizer, v::SupportedFunction, et::ET)
    return CI{SupportedFunction, ET}(vi.value)
end
