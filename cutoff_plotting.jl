""" Plotting Functions
"""

function get_plotting_strs(Exp::Experiment)
    """ Returns the strings needed for the plotting functions based on the Experiment.
    """
    @unpack_Experiment Exp

    diststr_greek = ""
    diststr_nongreek = ""
    actstr = ""
    dynamicsstr = ""

    ### Distribution
    if typeof(WeightDist) == Normal{Float64}
        μ = WeightDist.μ
        std = WeightDist.σ
        std_greek = std
        std_nongreek = std

        if std == 1/√N
            std_greek = "1/√N"
            std_nongreek = "sqrtN"
        end

        diststr_greek = "Gaussian($μ, $std_greek)"
        diststr_nongreek = "Gaussian($μ, $std_nongreek)"

    elseif typeof(WeightDist) == Uniform{Float64}
        a = WeightDist.a
        b = WeightDist.b

        diststr_greek = "Uniform($a, $b)"
        diststr_nongreek = "Uniform($a, $b)"

        if a == -1/√N && b == 1/√N
            diststr_greek = "Uniform(-1/√N, 1/√N)"
            diststr_nongreek = "Uniform(-sqrtN, sqrtN)"
        end
    end

    ### Activation
    if activation == σ
        actstr = "ReLu"
    elseif activation == tanh
        actstr = "TanH"
    end

    ### Forward vs Backward
    if forward
        dynamicsstr = "forward"
    else
        dynamicsstr = "backward"
    end

    return diststr_greek, diststr_nongreek, actstr, dynamicsstr
end

function println_hist(hist)
    """ Prints the probability of hitting time at time t. Assumes that the histogram starts at t=-1 and
        skips printing that.
    """
    for (idx, w) in enumerate(hist.weights)
        if idx == 1
            continue
        else
            println("Pr[T=" * string(idx-2) * "]: ", w)
        end
    end
end

function run_and_plot_tvds(Exp::Experiment, Results::ExperimentResults; verbose=false, save=false)
    """
    """
    @unpack_Experiment Exp

    run_chain(Exp, Results, verbose=verbose)
    diststr_greek, diststr_nongreek, actstr, dynamicsstr = get_plotting_strs(Exp)

    if activation == σ
        title = "$diststr_greek, $actstr, N=$N, \n $dynamicsstr, $num_chains chains"
    elseif activation == tanh
        title = "$diststr_greek, $actstr, N=$N, \n $dynamicsstr, $num_chains chains, $grid_size grid size"
    end

    p = plot()
    plot!(Results.tvds,
         title=title,
         xlabel="# layers",
         ylabel="tvd",
         xlim=(0, num_steps),
         ylim=(-0.2, 1.2),
         yticks = 0:0.2:1.2,
         seriestype=:scatter
    )
    if save
        savefig(p, "imgs/tvds/$diststr_nongreek $actstr $dynamicsstr N=$N.png")
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

function grid_size_effects(Exp::Experiment, grid_sizes)
    """
    """
    @unpack_Experiment Exp
    p = plot()

    for grid_size in grid_sizes
        println(grid_size)
        CurrExp = Experiment(X₀, N, num_chains, Dist, activation, grid_size, num_steps, forward, store_steps)
        Results = ExperimentResults([], [], [])

        run_chain(CurrExp, Results, verbose=false)
        diststr_greek, diststr_nongreek, actstr, dynamicsstr = get_plotting_strs(Exp)

        plot!(Results.tvds,
             title="$diststr_greek, $actstr, N=$N, \n $dynamicsstr, $num_chains chains",
             xlabel="# layers",
             ylabel="tvd",
             xlim=(0, num_steps),
             ylim=(-0.2, 1.2),
             yticks = 0:0.2:1.2,
             seriestype=:scatter,
            label="$grid_size"
        )
    end
    return p
end
