""" cutoff
"""

function tvd(μ, ϕ)
    """ Computes the total variation distance between the two probability dists
        μ and ϕ.
    """
    @assert length(μ) == length(ϕ)
    total_diff = 0

    for i = 1:length(μ)
        total_diff += abs(μ[i] - ϕ[i])
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

"""
TODO: The functions below are wrong!
"""
function ehrnfest(num_urns)
    """ Returns the transition matrix of the ehrnfest model on ``num_urns" urns.
    """
    P = zeros(num_urns, num_urns)
    n = num_urns - 1

    for i = 0:n
        for j = 0:n
            if j == i+1
                P[i+1, j+1] = (n-i)/n
            elseif j == i - 1
                P[i+1, j+1] = i/n
            end
        end
    end
    P
end

function lazy_ehrenfest(num_urns)
    """ Returns the transition matrix for the lazy ehrenfest model.
    """
    P = zeros(num_urns, num_urns)
    n = num_urns - 1

    for i = 0:n
        for j = 0:n
            if j == i-1
                P[i+1, j+1] = i/(n+1)
            elseif j == i
                P[i+1, j+1] = (n - i)/(n+1)
            end
        end
    end
    P
end
