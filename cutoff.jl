""" Library for cutoff experiments
"""

using StatsBase
using DataStructures
using Plots
using Distributions
using Parameters
using LinearAlgebra
using Random

include("structs.jl")
include("tvd.jl")
include("cutoff_plotting.jl")

""" Activation functions
"""
function σ(x::Real)
    """ The ReLu activation function.
    """
    if x > 0
        return x
    else
        return 0
    end
end


""" Markov Chain Functions
"""

function take_forward_step!(Exp::Experiment, Results::ExperimentResults, X, ϕ)
    """ Takes a random step using the a linear map generated from ``Dist" for each
        Xᵢ in X. Returns the probability distribution of Xᵢ at this step on a hypergrid
        of granularity ``grid_size".

        Note: This function overwrites the array ``X" at each step, in true Markov
        Chain fashion, and forgets the previous steps. Similarly, we do not currently
        keep track of the μ distribtions of previous steps.
    """
    @unpack_Experiment Exp

    μ = Dict{NTuple{N, Interval}, Float64}()
    weights = Array{Matrix{Float64}, 1}()
    steps = Array{Array{Float64}, 1}()

    for i=1:num_chains
        # take a step for the i'th chain
        Wᵢ = rand(Dist, N, N)
        step = activation.(Wᵢ * X[i])

        X[i] = step

        if store_steps
            push!(weights, Wᵢ)
            push!(steps, step)
        end

        # update the distribution of Xᵢs observed at this time step
        update_dist!(μ, X[i], grid_size)
    end
    # update Results
    normalize!(μ, num_chains)
    push!(Results.tvds, tvd(μ, ϕ))
    if store_steps
        push!(Results.weights, weights)
        push!(Results.steps, steps)
    end
end

function sample_weights!(Exp::Experiment, Results::ExperimentResults)
    """
    """
    @unpack_Experiment Exp
    weights = Array{Matrix{Float64}, 1}()

    for i=1:num_chains
        Wᵢ = rand(Dist, N, N)
        push!(weights, Wᵢ)
    end

    push!(Results.weights, weights)
end

function take_reverse_step!(Exp::Experiment, Results::ExperimentResults)
    """
    """
    @unpack_Experiment Exp
    steps = Array{Array{Float64}, 1}()

    for i = 1:num_chains
        step = X₀
        for j = length(Results.weights):-1:1
            step = activation.(Results.weights[j][i] * step)
        end
        push!(steps, step)
    end

    push!(Results.steps, steps)
end

function get_distribution_at_step(X::Array, N::Int, num_chains::Int, grid_size::Float64)
    """
    """
    μ = Dict{NTuple{N, Interval}, Float64}()

    for i=1:num_chains
        update_dist!(μ, X[i], grid_size)
    end

    normalize!(μ, num_chains)
    μ
end

function take_reverse_step!(Exp::Experiment, Results::ExperimentResults, ϕ)
    """
    """
    @unpack_Experiment Exp

    sample_weights!(Exp, Results)
    take_reverse_step!(Exp, Results)

    X = last(Results.steps)
    μ = get_distribution_at_step(X, N, num_chains, grid_size)
    push!(Results.tvds, tvd(μ, ϕ))
end

function initialize_chain_variables(Exp::Experiment)
    """
    """
    @unpack_Experiment Exp

    # the first step for every chain is the same.
    X = Array{Vector{Float64}, 1}()
    Xᵢ = X₀
    for i = 1:Exp.num_chains
        push!(X, X₀)
    end

    zero_coords = get_interval(zeros(N), grid_size)
    ϕ = Dict{NTuple{N, Interval}, Float64}()
    ϕ[zero_coords] = 1

    X, ϕ
end


function run_chain(Exp::Experiment, Results::ExperimentResults; verbose::Bool=false)
    """ Returns an Array of total variation distances between the distribution
        of X₀ and the point mass at 0.
    """
    X, ϕ = initialize_chain_variables(Exp)
    # make sure that this is the start of a new experiment
    @assert isempty(Results.weights)

    # Run chain
    for i = 1:Exp.num_steps
        if verbose
            println("Taking Step $i of $num_steps steps")
        end

        if forward
            take_forward_step!(Exp, Results, X, ϕ)
        else
            take_reverse_step!(Exp, Results, ϕ)
        end
    end
end


function hitting_times(Exp::Experiment)
    """ Calculates the hitting time to 0 for each sample path and returns them
        in an array.
    """
    @unpack_Experiment Exp
    times = Vector{Float64}()
    W = rand(Dist, N, N)
    num_exceeds = 0
    zero_interval = get_interval(zeros(N), grid_size)

    for i=1:num_chains
        X = X₀
        no_hit = true
        for t=1:num_steps
            # take a step
            rand!(Dist, W) # reuse the Weights matrix so we don't allocate new memory
            X = activation.(W * X) # rewrite X

            if activation == σ
                if iszero(X)
                    push!(times, t)
                    no_hit = false
                    break
                end
            elseif activation == tanh
                if zero_interval == get_interval(X, grid_size)
                    push!(times, t)
                    no_hit = false
                    break
                end
            end
        end
        if no_hit
            num_exceeds += 1
        end
    end

    println("$num_exceeds/$num_chains paths did not converge to 0 within $num_steps steps.")

    # this statement is so that normalization takes num_exceeds into account. a negative value is
    # used and the "hack" is to plot only from 0 and upwards, therefore covering the negative value's bar in the plot.
    for i in 1:num_exceeds
        push!(times, -10)
    end
    times
end
