""" Experiment Struct that has all the information for an experiment
    Note: using the @with_kw macro lets us use the @unpack macro.
"""
@with_kw struct Experiment
    X₀::Vector{Float64}
    N::Int
    num_chains::Int
    Dist::Distribution
    activation::Function
    grid_size::Float64
    num_steps::Int
    forward::Bool
    store_steps::Bool
end

mutable struct ExperimentResults
    tvds::Vector{Float64}
    weights::Array{Array{Matrix{Float64}}}
    steps::Array{Array{Array{Float64}}}
end

struct Interval
    left::Float64
    right::Float64
    Interval(left, right) = left > right ? error("Left element is greater than Right element.") : new(left, right)
end
