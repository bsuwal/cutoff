""" cutoff
"""

function tvd(μ, ϕ)
    """ Computes the total variation distance between the two probability dists
        μ and ϕ.
    """
    max_diff = 0
    for i = 1:length(μ)
        diff = abs(μ[i] - ϕ[i])

        if diff > max_diff
            max_diff = diff
        end
    end
    max_diff
end

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
