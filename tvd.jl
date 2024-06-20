""" Distance functions
"""

function tvd(μ::Dict, ϕ::Dict)::Float64
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

function get_interval(point, grid_size)
    """ Gets the interval that ``point" is in on a hypergrid of size ``grid_size".
    """
    intervals = Array{Interval, 1}()
    for coord in point
        interval = get_coordinate_interval(coord, grid_size)
        push!(intervals, interval)
    end
    Tuple(intervals)
end

function get_coordinate_interval(val, grid_size, digits = 8)::Interval
    """ Returns interval that ``val" is in on a hypergrid of size ``grid_size".
        Note: 0 is centered on [-grid_size/2, grid_size/2]. The rounding business in this
              function is to allow for this centering.
    """
    @assert grid_size >= 0.000000001 "smallest grid_size allowed is 0.000000001 (change the `digits' param to allow for smaller grid sizes.)"

    if val >= 0
        left = round(grid_size/2 + floor((val - grid_size/2)/grid_size) * grid_size, digits = digits)
        right = round(left + grid_size, digits = digits)
    else
        right = round(-grid_size/2 + ceil((val + grid_size/2)/grid_size) * grid_size, digits = digits)
        left = round(right - grid_size, digits = digits)
    end

    Interval(left, right)
end

function update_dist!(μ::Dict, X, grid_size::Float64)
    """ Adds 1 to the observation of state ``X" to the distribution
        μ (which is a dictionary).
    """
    interval = get_interval(X, grid_size)

    if interval in keys(μ)
        μ[interval] += 1
    else
        μ[interval] = 1
    end
end
