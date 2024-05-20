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

function σ²(x::Real)
    """ The ReQu activation function.
    """
    if x > 0
        return x^2
    else
        return 0
    end
end

""" Cutoff functions
"""

function tvd(μ::Array, ϕ::Array)
    """ Computes the total variation distance between the distributions μ and ϕ.
    """
    @assert size(μ) == size(ϕ)
    total_diff = 0

    for i in eachindex(μ)
        total_diff += abs(μ[i] - ϕ[i])
    end

    total_diff/2
end

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

function num_balls_distribution(num_balls, urn_counts)
    """ Returns the distribution of the number of balls present in the urn
        corresponding to ``urn_counts". The i'th index of the returned array ``dist"
        contains the number of times the urn had i balls in it.
    """
    c = counter(urn_counts)
    denom = sum(values(c))

    dist = zeros(num_balls)
    for i=1:num_balls
        if i in keys(c)
            dist[i] = c[i] / denom
        end
    end
    dist
end

function run_chain(balls, num_steps, lazy=false, urn=1)
    """ Returns an array that holds the counts of the number of balls in
        urn ``urn".
    """
    num_urns = 2
    num_balls = sum(balls)
    @assert urn <= num_urns

    urn_counts = [] # holds the number of balls in urn ``urn"

    for i = 1:num_steps
        # pick source of ball to move
        src = StatsBase.sample(1:num_urns, ProbabilityWeights(balls))

        dst = get_destination(src, lazy, num_balls)

        balls[src] -= 1
        balls[dst] += 1

        push!(urn_counts, balls[urn])
    end
    urn_counts
end

function get_destination(src, lazy, n)
    """ Returns the destination of the ball for a step in the ehrenfest chain.
        For the simple model, the destination is the other urn w.p 1.
        For the lazy model, there is a (1/(n+1)) probability that the ball
        remains in the same urn.
    """
    if src == 1
        dst = 2
    elseif src == 2
        dst = 1
    end

    if lazy && rand() < (1/(n+1))
        dst = src
    end

    dst
end

function normalize_dictionary(d::Dict)
    """ Normalizes the dictionary using L1 norm.
        Important: The function assumes that the values of ``d" are non-negative.
    """
    denom = sum(values(d))
    normalized = Dict()

    for (key, value) in d
        normalized[key] = value/denom
    end

    normalized
end

# function normalize_dictionary(ds::Array{Dict})
#     """ Normalizes the dictionary using L1 norm.
#         Important: The function assumes that the values of ``d" are non-negative.
#     """
#     normalized_dicts = []
#
#     for i = 1:length(ds)
#         normalized = normalize_dictionary(ds[i])
#         push!(normalized_dicts, normalized)
#     end
#
#     normalized_dicts
# end

function get_interval(point, step_size)
    """ Gets the interval that point is in on a hypergrid of size ``step_size".
    """
    intervals = []
    for coord in point
        interval = get_coordinate_interval(coord, step_size)
        push!(intervals, interval)
    end
    Tuple(intervals)
end

function get_coordinate_interval(val, step_size)
    """ Returns interval that ``val" is in on a hypergrid of size ``step_size".
        Note: 0 is centered on [-step_size/2, step_size/2].
    """
    if val >= 0
        left = round(step_size/2 + floor((val - step_size/2)/step_size) * step_size, digits = 3)
        right = round(left + step_size, digits=3)
    else
        right = round(-step_size/2 + ceil((val + step_size/2)/step_size) * step_size, digits = 3)
        left = round(right - step_size, digits=3)
    end

    left, right
end

function take_step(Xᵢ, Dist, activation::Function, N::Int)
    """ Takes a random step using the a linear map generated from ``Dist"
        and returns the step.
    """
    Wᵢ = rand(Dist, N, N)
    Xᵢ₊₁= activation.(Wᵢ * Xᵢ)
    Xᵢ₊₁
end

# function take_step(X::Array{Vector{Float64}}, Dist, activation::Function, N::Int)
#     """ Takes a random step using the a linear map generated from ``Dist"
#         and returns the step.
#     """
#     X_arr = []
#     for i = 1:length(X)
#         Xᵢ₊₁ = take_step(X[i], Distm activation, N)
#         push!(X_arr, Xᵢ₊₁)
#     end
#     X_arr
# end

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

# function update_dist!(μs::Array{Dict}, X, step_size::Float)
#     """ Adds 1 to the observation of state ``X" to the distribution
#         μ (which is a dictionary).
#     """
#     for i = 1:length(X)
#         update_dist!(μ, X[i], step_size)
#     end
# end

function get_tvds(X₀, Dist, activation, num_steps, N, step_size)
    """ Returns an Array of total variation distances between the distribution
        of X₀ and the point mass at 0.
    """
    Xᵢ = X₀
    μ = Dict()
    tvds = []

    zero_coords = get_interval(zeros(N), step_size)
    ϕ = Dict()
    ϕ[zero_coords] = 1

    for i=1:num_steps
        Xᵢ₊₁ = take_step(Xᵢ, Dist, activation, N)
        update_dist!(μ, Xᵢ₊₁, step_size)

        μ_normalized = normalize_dictionary(μ)

        push!(tvds, tvd(μ_normalized, ϕ))
        Xᵢ = Xᵢ₊₁
    end
    tvds
end

# function get_tvds(X₀, Dist, activation, num_steps, N, step_size, num_chains::Integer)
#     """ Returns an Array of total variation distances between the distribution
#         of X₀ and the point mass at 0.
#     """
#     X = []
#     Xᵢ = X₀
#     μs = Array{Dict}(undef, num_chains)
#     for i = 1:num_chains
#         μs[i] = Dict()
#         push!(X, X₀)
#     end
#     tvds = []
#
#     zero_coords = get_interval(zeros(N), step_size)
#     ϕ = Dict()
#     ϕ[zero_coords] = 1
#
#     for i = 1:num_steps
#         Xᵢ₊₁ = take_step(Xᵢ, Dist, activation, N)
#         update_dist!(μs, Xᵢ₊₁, step_size)
#
#         μ_normalized = normalize_dictionary(μs)
#
#         push!(tvds, tvd(μ_normalized, ϕ))
#         Xᵢ = Xᵢ₊₁
#     end
#     tvds
# end


"""
TODO: The functions below are wrong!
"""
nothing
# function ehrnfest(num_urns)
#     """ Returns the transition matrix of the ehrnfest model on ``num_urns" urns.
#     """
#     P = zeros(num_urns, num_urns)
#     n = num_urns - 1
#
#     for i = 0:n
#         for j = 0:n
#             if j == i+1
#                 P[i+1, j+1] = (n-i)/n
#             elseif j == i - 1
#                 P[i+1, j+1] = i/n
#             end
#         end
#     end
#     P
# end
#
# function lazy_ehrenfest(num_urns)
#     """ Returns the transition matrix for the lazy ehrenfest model.
#     """
#     P = zeros(num_urns, num_urns)
#     n = num_urns - 1
#
#     for i = 0:n
#         for j = 0:n
#             if j == i-1
#                 P[i+1, j+1] = i/(n+1)
#             elseif j == i
#                 P[i+1, j+1] = (n - i)/(n+1)
#             end
#         end
#     end
#     P
# end
