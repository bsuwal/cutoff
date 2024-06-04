""" Library for cutoff experiments
"""

using StatsBase
using DataStructures
using Plots
using Distributions
using Parameters
using LinearAlgebra

""" Experiment Struct that has all the information for an experiment
    Note: using the @with_kw macro lets us use the @unpack macro.
"""
@with_kw struct Experiment
    X₀::Array{Float64, 1}
    N::Int
    num_chains::Int
    Dist::Distribution
    activation::Function
    step_size::Float64
    num_steps::Int
    forward::Bool
end



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

""" Distance functions
"""

function tvd(μ::Dict, ϕ::Dict)
    """ Computes the total variation distance between the distributions μ and ϕ, where
        μ and ϕ are dictionarys.
    """
    total_diff = 0

    intersection = intersect(keys(μ), keys(ϕ))

    for key in intersection
        total_diff += abs(μ[key] - ϕ[key])
    end

    for key in setdiff(keys(μ), keys(ϕ))
        total_diff += abs(μ[key])
    end

    for key in setdiff(keys(ϕ), keys(μ))
        total_diff += abs(ϕ[key])
    end

    total_diff/2
end

""" Functions to compute the intervals in the hypergrid.
"""

function normalize!(d::Dict, denom)
    """ Normalizes ``d" with the denominator ``denom".
        Important: The function assumes that the values of ``d" are non-negative.
    """
    for (key, value) in d
        d[key] = value/denom
    end
end

function get_interval(point, step_size)
    """ Gets the interval that ``point" is in on a hypergrid of size ``step_size".
    """
    intervals = []
    for coord in point
        interval = get_coordinate_interval(coord, step_size)
        push!(intervals, interval)
    end
    Tuple(intervals)
end

function get_coordinate_interval(val, step_size, digits = 4)
    """ Returns interval that ``val" is in on a hypergrid of size ``step_size".
        Note: 0 is centered on [-step_size/2, step_size/2]. The rounding business in this
              function is to allow for this centering.
    """
    @assert step_size >= 0.001 "smallest step_size allowed is 0.001 (change the `digits' param to allow for smaller step sizes.)"

    if val >= 0
        left = round(step_size/2 + floor((val - step_size/2)/step_size) * step_size, digits = digits)
        right = round(left + step_size, digits = digits)
    else
        right = round(-step_size/2 + ceil((val + step_size/2)/step_size) * step_size, digits = digits)
        left = round(right - step_size, digits = digits)
    end

    left, right
end

""" Markov Chain Functions
"""

function take_step!(X, Dist, activation::Function, N::Int, num_chains::Int,
    step_size::Float64, all_weights, all_steps; store_steps::Bool=false)
    """ Takes a random step using the a linear map generated from ``Dist" for each
        Xᵢ in X. Returns the probability distribution of Xᵢ at this step on a hypergrid
        of granularity ``step_size".

        Note: This function overwrites the array ``X" at each step, in true Markov
        Chain fashion, and forgets the previous steps. Similarly, we do not currently
        keep track of the μ distribtions of previous steps.
    """
    μ = Dict()
    weights = []
    steps = []

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
        update_dist!(μ, X[i], step_size)
    end

    if store_steps
        push!(all_weights, weights)
        push!(all_steps, steps)
    end

    normalize!(μ, num_chains)
    μ
end

function sample_weights!(Dist, N::Int, num_chains::Int, all_weights::Array)
    """
    """
    weights = []

    for i=1:num_chains
        # take a step for the i'th chain
        Wᵢ = rand(Dist, N, N)
        push!(weights, Wᵢ)
    end

    push!(all_weights, weights)
end

function take_reverse_step!(X₀, all_weights::Array, all_steps::Array, num_chains::Int, activation)
    """
    """
    steps = []
    for i = 1:num_chains
        step = X₀
        for j = length(all_weights):-1:1
            step = activation.(all_weights[j][i] * step)
        end
        push!(steps, step)
    end
    push!(all_steps, steps)
end

function get_distribution_at_step(X::Array, num_chains, step_size)
    """
    """
    μ = Dict()

    for i=1:num_chains
        update_dist!(μ, X[i], step_size)
    end

    normalize!(μ, num_chains)
    μ
end

function take_reverse_step!(X₀::Array, Dist, activation::Function, N::Int, num_chains::Int,
    step_size::Float64, all_weights::Array, all_steps::Array)
    """ Takes a random step using the a linear map generated from ``Dist" for each
        Xᵢ in X. Returns the probability distribution of Xᵢ at this step on a hypergrid
        of granularity ``step_size".

        Note: This function overwrites the array ``X" at each step, in true Markov
        Chain fashion, and forgets the previous steps. Similarly, we do not currently
        keep track of the μ distribtions of previous steps.
    """
    sample_weights!(Dist, N, num_chains, all_weights)
    take_reverse_step!(X₀, all_weights, all_steps, num_chains, activation)

    X = last(all_steps)
    μ = get_distribution_at_step(X, num_chains, step_size)
    μ
end


function update_dist!(μ::Dict, X, step_size::Float64)
    """ Adds 1 to the observation of state ``X" to the distribution
        μ (which is a dictionary).
    """
    interval = get_interval(X, step_size)

    if interval in keys(μ)
        μ[interval] += 1
    else
        μ[interval] = 1
    end
end

function mc_tvds(Exp::Experiment; verbose::Bool=false, store_steps::Bool=false)
    # exp = Experiment(X₀, N, num_chains, Dist, activation, step_size, num_steps)
    """ Returns an Array of total variation distances between the distribution
        of X₀ and the point mass at 0.
    """
    @unpack_Experiment Exp
    # Initialize parameters
    tvds = []

    # the first step for every chain is the same.
    X = []
    Xᵢ = X₀
    for i = 1:num_chains
        push!(X, X₀)
    end

    zero_coords = get_interval(zeros(N), step_size)
    ϕ = Dict()
    ϕ[zero_coords] = 1
    all_weights = []
    all_steps = []

    # Run chain
    for i = 1:num_steps
        if verbose
            println("Taking Step $i of $num_steps steps")
        end

        if forward
            μ = take_step!(X, Dist, activation, N, num_chains, step_size, all_weights, all_steps, store_steps=true)
        else
            μ = take_reverse_step!(X₀, Dist, activation, N, num_chains, step_size, all_weights, all_steps)
        end
        push!(tvds, tvd(μ, ϕ))
    end

    if store_steps
        return tvds, all_weights, all_steps
    else
        return tvds
    end
end

function time_to_convergence_to_zero(Exp::Experiment, num_paths)
    @assert Exp.num_chains == 1
    times = []
    num_exceeds = 0

    for i=1:num_paths
        tvds = mc_tvds(Exp, verbose=false, store_steps=false)
        time = findfirst(==(0), tvds)
        if isnothing(time)
            num_exceeds += 1
        else
            push!(times, time)
        end
    end

    num_steps = Exp.num_steps
    println("$num_exceeds/$num_paths paths did not converge to 0 within $num_steps steps.")
    times
end


""" Plotting Functions
"""

function get_plotting_strs(Exp::Experiment)
    diststr = ""
    actstr = ""

    if typeof(Exp.Dist) == Normal{Float64}
        μ = Exp.Dist.μ
        std = Exp.Dist.σ

        if std == 1/√Exp.N
            std = "1/√N"
        end
        diststr = "Gaussian($μ, $std)"
    elseif typeof(Exp.Dist) == Uniform{Float64}
        a = Exp.Dist.a
        b = Exp.Dist.b

        if a == -1/√Exp.N
            a = "-1/√N"
        end
        if b == 1/√Exp.N
            b = "1/√N"
        end
        diststr = "Uniform($a, $b)"
    end

    if Exp.activation == σ
        actstr = "ReLu"
    elseif Exp.activation == tanh
        actstr = "TanH"
    end

    return diststr, actstr
end

function run_and_plot_tvd_experiment(Exp::Experiment; verbose=false, save=false)
    """
    """
    @unpack_Experiment Exp

    tvds = mc_tvds(Exp, verbose=verbose)
    diststr, actstr = get_plotting_strs(Exp)

    p = plot()
    plot!(tvds,
         title="$diststr, $actstr, N=$N, $num_chains chains",
         xlabel="# layers",
         ylabel="tvd",
         xlim=(0, num_steps),
         ylim=(-0.2, 1.2),
         yticks = 0:0.2:1.2,
         seriestype=:scatter
    )
    if save
        savefig(p, "$diststr $actstr N=$N.png")
    else
        display(p)
    end
end

function plot_single_coordinate_over_time(Exp::Experiment, all_steps, coord)
    """
    """
    @unpack_Experiment Exp

    # plotting setup
    xs = []
    ys = []
    for step in eachindex(all_steps)
        for chain in all_steps[step]
            push!(xs, step)
            push!(ys, chain[coord])
        end
    end

    diststr, actstr = get_plotting_strs(Exp)

    # the plotting
    println()
    println("X₀ : $X₀")
    plot(xs, ys,
        title="$diststr, $actstr, N=$N, $num_chains chains",
        xlabel="# layers",
        ylabel="Xᵢ",
        seriestype=:scatter,
        xlim=(0, num_steps),
        ylim=(-1.2, 1.2),
        yticks = -1.2:0.2:1.2,
    )
end
