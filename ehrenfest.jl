
""" Functions for the ehrenfest simulation.
"""

function transition_matrix_lazy_ehrenfest(n::Int)
    """ Constructs the transition matrix P for the lazy ehrenfest urn model.
        P is of dims (n+1, n+1) - n+1 because the urn can also have 0 balls.
    """
    P = zeros(n+1, n+1) # n+1 because the urn can also have 0 balls

    for i = 0:n
        j = i+1 # the actual matrix index

        if j != 1
            P[j, j-1] = i/(n+1)
        end

        P[j, j] = 1/(n+1)

        if j!=n+1
            P[j, j+1] = (n-i)/(n+1)
        end
    end

    P
end

function stationary_distribution(P::Matrix{Float64}, sigdigits::Int=4)
    """ Generates the stationary distribution associated with the transition matrix P
        by generating a vector π s.t πP = π.

        ``sigdigits" is the number of significant digits up to which π needs to equal to
        its last iteration elementwise to assert that π has stopped changing.
    """
    is_approx_arrays(a, b, precision) = round.(a, sigdigits=precision) == round.(b, sigdigits=precision)

    prev = zeros(size(P)[1])'     # initialize vector (any vector that sums up to 1 will do)
    prev[1] = 1

    while true
        curr = prev * P
        if is_approx_arrays(curr, prev, sigdigits)
            return curr'
        end
        prev = curr
    end
end

""" Functions to run Markov Chain instances.
"""
function initialize_ball_assignments(dist::Vector{Int})
    """ Creates and returns a Dictionary where the keys are the balls and
        the values are the urns that the balls are in.

        Argument: A Vector ``dist" of length 2.
                  dist[1] is the number of balls in urn 1 and
                  dist[2] is the number of balls in urn 2.
    """
    @assert length(dist) == 2

    assignments = Dict{Int, Int}()
    num_balls = dist[1]+dist[2]

    for ball = 1:dist[1]
        assignments[ball] = 1
    end
    for ball = dist[1]+1:num_balls
        assignments[ball] = 2
    end

    @assert length(assignments) == dist[1] + dist[2]
    assignments
end

function initialize_ball_assignments!(assignments::Dict{Int, Int}, dist::Vector{Int})
    """ Returns a dictionary ``assignments" where the keys are the balls and
        the values are the urns that the balls are in. The ``assignments" dict
        is passed in as an argument and overwritten - the intention is that
        we don't create a new Dict and therefore save time by not creating a
        new object.

        Argument: A Vector ``dist" of length 2.
                  dist[1] is the number of balls in urn 1 and
                  dist[2] is the number of balls in urn 2.

                  a Dict ``assignments" that gets overwritten.
    """
    @assert length(dist) == 2
    num_balls = dist[1]+dist[2]

    for ball = 1:dist[1]
        assignments[ball] = 1
    end
    for ball = dist[1]+1:num_balls
        assignments[ball] = 2
    end
    @assert length(assignments) == dist[1] + dist[2]
end

function take_step!(assignments::Dict{Int, Int}, balls::Vector{Int})::Int
    """ Changes the assignments Dict to put the ball in the other urn.
        Returns -1 if ball was put into urn 2 from urn 1.
        Returns +1 if ball was put into urn 1 from urn 2.
    """
    # choose a ball to put in the other urn
    ball = rand(balls)

    # this happens only if the chain takes lazy steps
    if ball == -1
        return 0
    end

    if assignments[ball] == 1
        assignments[ball] = 2
        return -1
    elseif assignments[ball] == 2
        assignments[ball] = 1
        return 1
    else
        error("Ball Assignment must be 1 or 2")
    end
end

function run_chain!(num_steps::Int, assignments::Dict{Int, Int},
                    balls::Vector{Int}, initial_urn_count::Int)
    """ Returns an array that holds the counts of the number of balls in
        urn 1 at each time step.

        Args: initial_dist - the Vector that holds the distribution of balls in
                             each urn, for eg [256, 256]
    """
    urn_counts = Vector{Int}() # holds the number of balls in urn 1 at each time step
    urn_count = initial_urn_count

    for i = 1:num_steps
        urn_count += take_step!(assignments, balls)
        push!(urn_counts, urn_count)
    end
    urn_counts
end

function run_chains(initial_dist::Vector{Int}, num_steps::Int, num_chains::Int;
        lazy::Bool=false)
    """ Runs ``num_chains" chains for ``num_steps" steps. Each chain instance
        starts with the initial distributions ``initial dist" which is a
        Vector of length 2. initial_dist[1] is the number of balls in urn 1
        and initial_dist[2] is the number of balls in urn 2.

        Returns a Vector that contains an array of length ``num_chains" where
        the i'th entry is an array of length ``num_steps" that contains
        the number of balls in urn 1 for that chain instance.
    """
    if !lazy
        println("Warning: Non-lazy instances of the chain are being generated,
                 but Aldous only proved cutoff for the lazy version of the chain.")
    end

    all_counts = Vector{Vector{Int}}()
    num_balls = sum(initial_dist)

    # Reusable containers so we don't allocate too much memory
    assignments = initialize_ball_assignments(initial_dist)
    balls = collect(1:num_balls)
    if lazy
        push!(balls, -1)
    end

    for i = 1:num_chains
        initialize_ball_assignments!(assignments, initial_dist)

        urn_counts = run_chain!(num_steps, assignments, balls, initial_dist[1])
        push!(all_counts, urn_counts)
    end

    all_counts
end

function get_distribution_at_time_step(all_counts::Vector{Vector{Int}}, t::Int)
    """ Returns an array that holds the number of balls in urn 1 at time step t.
    """
    dist = Vector{Int}()
    for i = 1:length(all_counts)
        push!(dist, all_counts[i][t])
    end
    dist
end

function expected_hitting_times_from_0(n::Int)
    """ Computes the expected hitting times for the lazy ehrenfest model
        when starting at (0, n).

        ``k" is the number of balls in urn 1.

        Returns a vector of length ``n" where the i'th entry is the expected hitting time
        for the chain to hit state i.
    """
    times = Vector{Float64}()

    curr = 0
    for i=0:n-1
        curr += (n+1)/(n-i)
        push!(times, curr)
    end

    times
end

function get_tvds(all_counts, μ)
    """ Obtains the total variation distance.
        μ is the stationary distribution.
    """
    num_steps = length(all_counts[1]) # we assume each chain has the same number of steps

    tvds = Vector{Float64}()
    curr_tvd = 0
    for step = 1:num_steps
        dist = get_distribution_at_time_step(all_counts, step)

        bins = 0:1:num_balls+1
        ϕ = normalize(fit(Histogram, dist, bins), mode=:probability)
        curr_tvd = tvd(μ, ϕ.weights)

        push!(tvds, curr_tvd)
    end

    tvds
end
