""" Library for cutoff experiments
"""

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

function take_step!(X::Array, Dist, activation::Function, N::Int, num_chains::Int, step_size::Float64)
    """ Takes a random step using the a linear map generated from ``Dist" for each
        Xᵢ in X. Returns the probability distribution of Xᵢ at this step on a hypergrid
        of granularity ``step_size".

        Note: This function overwrites the array ``X" at each step, in true Markov
        Chain fashion, and forgets the previous steps. Similarly, we do not currently
        keep track of the μ distribtions of previous steps.
    """
    μ = Dict()
    for i=1:num_chains
        # take a step for the i'th chain
        Wᵢ = rand(Dist, N, N)
        X[i] = activation.(Wᵢ * X[i])

        # update the distribution of Xᵢs observed at this time step
        update_dist!(μ, X[i], step_size)
    end

    normalize!(μ, num_chains)
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

function mc_tvds(X₀, Dist, activation, num_steps, N, step_size, num_chains::Integer)
    """ Returns an Array of total variation distances between the distribution
        of X₀ and the point mass at 0.
    """

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

    # Run chain
    for i = 1:num_steps
        μ = take_step!(X, Dist, activation, N, num_chains, step_size)
        push!(tvds, tvd(μ, ϕ))
    end
    tvds
end
