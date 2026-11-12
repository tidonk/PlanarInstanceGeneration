#!/usr/bin/env julia

using TopologyOptimizer
using JSON
using TOML


const APPLICATION = "DC_SCOTS_TIME_SLACK_BENDERS"


function require_setting(settings, keys, expected)::Nothing
    value = settings
    for key in keys
        value = get(value, key, nothing)
        value === nothing && error("missing required setting $(join(keys, '.'))")
    end
    value == expected || error(
        "setting $(join(keys, '.')) must be $(repr(expected)); got $(repr(value))"
    )
    return nothing
end


function validate_settings(settings)::Int
    require_setting(settings, ["APPLICATION"], APPLICATION)
    tal = get(get(get(settings, "LIMITS", Dict()), "BRANCH", Dict()), "TAL_TIMESTEP", nothing)
    tal isa Integer && tal >= 0 || error(
        "setting LIMITS.BRANCH.TAL_TIMESTEP must be a nonnegative integer; got $(repr(tal))"
    )
    require_setting(settings, ["LIMITS", "BRANCH", "BRANCH_SWITCHING"], true)
    require_setting(settings, ["LIMITS", "TRANSFORMER", "TRANSFORMER_SHIFTING"], false)
    require_setting(settings, ["LIMITS", "GENERATOR", "GEN_SWITCHING"], false)
    require_setting(settings, ["SUBSTATION_RECONF", "BRANCH_RECONF"], false)
    require_setting(settings, ["SUBSTATION_RECONF", "COUPLER_RECONF"], false)
    require_setting(settings, ["LOSS_ESTIMATION", "ESTIMATE_LOSSES"], false)
    require_setting(settings, ["POSTPROCESSING", "AC_FEAS", "CHECK"], false)
    if settings["WARMSTARTING"]["ENABLED"]
        require_setting(settings, ["WARMSTARTING", "WARMSTART_METHOD"], ":scopf")
        require_setting(settings, ["SOLVER", "INCLUDE_WARMSTART_TIME_LIMIT"], false)
    end
    require_setting(settings, ["HEURISTIC_SOLVING", "ENABLED"], false)
    return Int(tal)
end


function assert_loaded_invariants(expected_tal::Int)::Nothing
    TopologyOptimizer.PARAMS[:LIMITS][:BRANCH][:TAL_TIMESTEP] == expected_tal || error(
        "loaded TAL does not match the requested value $expected_tal"
    )
    TopologyOptimizer.PARAMS[:LIMITS][:BRANCH][:BRANCH_SWITCHING] || error("branch variable layer is disabled")
    !TopologyOptimizer.PARAMS[:LIMITS][:TRANSFORMER][:TRANSFORMER_SHIFTING] || error("transformer switching is enabled")
    !TopologyOptimizer.PARAMS[:LIMITS][:GENERATOR][:GEN_SWITCHING] || error("generator switching is enabled")
    !TopologyOptimizer.PARAMS[:SUBSTATION_RECONF][:BRANCH_RECONF] || error("branch reconfiguration is enabled")
    !TopologyOptimizer.PARAMS[:SUBSTATION_RECONF][:COUPLER_RECONF] || error("coupler reconfiguration is enabled")
    !TopologyOptimizer.PARAMS[:LOSS_ESTIMATION][:ESTIMATE_LOSSES] || error("loss estimation is enabled")
    !TopologyOptimizer.PARAMS[:POSTPROCESSING][:AC_FEAS][:CHECK] || error("AC feasibility checking is enabled")
    return nothing
end


function json_safe(value)
    if value === nothing || value isa Bool || value isa Integer || value isa AbstractString
        return value
    elseif value isa AbstractFloat
        return isfinite(value) ? Float64(value) : string(value)
    elseif value isa Symbol
        return string(value)
    elseif value isa AbstractDict
        return Dict(string(key) => json_safe(item) for (key, item) in value)
    elseif value isa AbstractVector || value isa Tuple
        return [json_safe(item) for item in value]
    end
    return string(value)
end


function redirected_directories(log_directory::String)
    return Dict{Symbol,Any}(
        :USER_ENVIRONMENT => Dict{Symbol,Any}(
            :DIRECTORIES => Dict{Symbol,Any}(
                :SLACK_CACHE_DIR => log_directory,
                :SLACKED_CASES_DIR => log_directory,
                :SOL_DIR => log_directory,
                :PLOT_DIR => log_directory,
                :INSTANCE_GEN_DIR => log_directory,
                :SOLVER_LOG_DIR => log_directory,
                :LOG_DIR => log_directory,
                :RESULTS_DIR => log_directory,
            ),
        ),
    )
end


function reset_innoptem_logger!()::Nothing
    memento = TopologyOptimizer.Memento
    empty!(memento.gethandlers(TopologyOptimizer._LOGGER))
    push!(
        TopologyOptimizer._LOGGER,
        memento.DefaultHandler(
            stdout,
            memento.DefaultFormatter("{level}: {msg}"),
            Dict{Symbol,Any}(:is_colorized => false),
        ),
    )
    return nothing
end


function interpolated_quantile(values::Vector{Float64}, probability::Float64)
    isempty(values) && return nothing
    ordered = sort(values)
    position = 1 + (length(ordered) - 1) * probability
    lower = floor(Int, position)
    upper = ceil(Int, position)
    if lower == upper
        return ordered[lower]
    end
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
end


function coupling_rating_validation(
    result,
    normalization,
    has_finite_objective::Bool,
    base_mva::Float64,
    network,
)
    changed = get(normalization, "changed_branches", [])
    changed_ids = Set(Int(item["row_one_based"]) for item in changed)
    coupling_rows = get(
        normalization, "coupling_branch_rows_one_based", collect(changed_ids)
    )
    coupling_ids = Set(Int(row) for row in coupling_rows)
    bound = Float64(get(normalization, "replacement_rating_mva", 0.0))
    observations = 0
    per_branch = Dict{String,Any}()
    noncoupling_per_branch = Dict{String,Float64}()
    branch_solution = get(
        get(get(get(result, "solution", Dict()), "nw", Dict()), "1", Dict()),
        "branch",
        Dict(),
    )
    for (key, values) in branch_solution
        match_value = match(r"^\(?\s*(\d+)", string(key))
        match_value === nothing && continue
        branch_id = parse(Int, match_value.captures[1])
        branch_flow = nothing
        for field in ("pf", "pt")
            value = get(values, field, nothing)
            if value isa Real && isfinite(value)
                flow = abs(Float64(value)) * base_mva
                branch_flow = branch_flow === nothing ? flow : max(branch_flow, flow)
                branch_id ∈ coupling_ids && (observations += 1)
            end
        end
        branch_flow === nothing && continue
        branch_key = string(branch_id)
        if branch_id ∈ coupling_ids
            per_branch[branch_key] = branch_flow
        else
            noncoupling_per_branch[branch_key] = branch_flow
        end
    end
    branch_flows = Float64[value for value in values(per_branch)]
    noncoupling_branch_flows = Float64[value for value in values(noncoupling_per_branch)]
    maximum_flow = isempty(branch_flows) ? nothing : maximum(branch_flows)
    changed_branch_flows = Float64[
        value for (key, value) in per_branch if parse(Int, key) ∈ changed_ids
    ]
    normalized_maximum_flow = isempty(changed_branch_flows) ?
        nothing : maximum(changed_branch_flows)
    maximum_utilization = normalized_maximum_flow !== nothing && bound > 0 ?
        normalized_maximum_flow / bound : nothing
    nonbinding_threshold = 0.95
    positive_flow_threshold_mw = 1.0e-6
    flowing_branch_count = count(flow -> flow > positive_flow_threshold_mw, branch_flows)
    observed_branch_count = length(branch_flows)
    coupling_branch_count = length(coupling_ids)
    network_branch_count = length(get(network, "branch", Dict()))
    noncoupling_branch_count = max(0, network_branch_count - coupling_branch_count)
    complete_normalized_flow_observation =
        length(changed_branch_flows) == length(changed_ids)
    empirically_nonbinding = if isempty(changed)
        true
    elseif !complete_normalized_flow_observation
        nothing
    else
        maximum_utilization !== nothing &&
            maximum_utilization < nonbinding_threshold
    end
    return Dict{String,Any}(
        "coupling_branch_count" => coupling_branch_count,
        "flow_unit" => "MW",
        "base_mva" => base_mva,
        "observed_coupling_branch_count" => observed_branch_count,
        "observed_coupling_branch_fraction" => coupling_branch_count > 0 ?
            observed_branch_count / coupling_branch_count : nothing,
        "flow_observation_count" => observations,
        "positive_flow_threshold_mw" => positive_flow_threshold_mw,
        "positive_flow_branch_count" => flowing_branch_count,
        "positive_flow_branch_fraction" => observed_branch_count > 0 ?
            flowing_branch_count / observed_branch_count : nothing,
        "mean_absolute_flow_mw" => observed_branch_count > 0 ?
            sum(branch_flows) / observed_branch_count : nothing,
        "median_absolute_flow_mw" => interpolated_quantile(branch_flows, 0.5),
        "p95_absolute_flow_mw" => interpolated_quantile(branch_flows, 0.95),
        "maximum_absolute_flow_mw" => maximum_flow,
        "noncoupling_branch_count" => noncoupling_branch_count,
        "observed_noncoupling_branch_count" => length(noncoupling_branch_flows),
        "observed_noncoupling_branch_fraction" => noncoupling_branch_count > 0 ?
            length(noncoupling_branch_flows) / noncoupling_branch_count : nothing,
        "noncoupling_mean_absolute_flow_mw" => isempty(noncoupling_branch_flows) ?
            nothing : sum(noncoupling_branch_flows) / length(noncoupling_branch_flows),
        "maximum_utilization" => maximum_utilization,
        "normalized_branch_count" => length(changed_ids),
        "observed_normalized_branch_count" => length(changed_branch_flows),
        "maximum_absolute_flow_by_branch_mw" => per_branch,
        "nonbinding_threshold" => nonbinding_threshold,
        "empirically_nonbinding" => empirically_nonbinding,
        "validation_status" => if isempty(changed)
            "not_applicable"
        elseif complete_normalized_flow_observation
            "evaluated_from_primal_solution"
        elseif has_finite_objective
            "not_observable_without_primal_flows"
        else
            "not_observable_without_primal_solution"
        end,
    )
end


function main(args)::Nothing
    length(args) == 4 || error(
        "usage: run_scopf_benders.jl INSTANCE SETTINGS RESULT LOG_DIRECTORY"
    )
    instance_file, settings_file, result_file, log_directory = abspath.(args)
    dirname(result_file) == log_directory || error("result must be inside LOG_DIRECTORY")
    isfile(instance_file) || error("instance does not exist: $instance_file")
    isfile(settings_file) || error("settings do not exist: $settings_file")
    mkpath(log_directory)

    settings = TOML.parsefile(settings_file)
    topological_action_limit = validate_settings(settings)
    normalization_file = joinpath(log_directory, "normalization.json")
    isfile(normalization_file) || error("normalization record does not exist")
    normalization = JSON.parsefile(normalization_file)

    solve_started = time_ns()
    reset_innoptem_logger!()
    data, result = TopologyOptimizer.main(
        instance_file;
        application=APPLICATION,
        slack_types=[:ratea],
        busbars=false,
        timesteps=1,
        plot=false,
        params_file=settings_file,
        additional_consts=redirected_directories(log_directory),
        return_data=true,
    )
    wall_seconds = (time_ns() - solve_started) / 1.0e9
    assert_loaded_invariants(topological_action_limit)

    network = data["nw"]["1"]
    base_mva = Float64(network["baseMVA"])
    contingencies = network["contingencies"]
    single_branch_candidates = count(
        branch -> get(branch, "type", "") ∉ ["breaker", "K"],
        values(network["branch"]),
    )
    retained_failures = count(
        contingency -> haskey(contingency, :branch) && length(contingency[:branch]) == 1,
        contingencies,
    )
    base_case_entries = count(
        contingency -> haskey(contingency, :branch) && isempty(contingency[:branch]),
        contingencies,
    )
    objective_value = get(result, "objective", nothing)
    has_finite_objective = objective_value isa Real && isfinite(objective_value)
    solution_networks = get(get(result, "solution", Dict()), "nw", nothing)
    has_primal_network = solution_networks isa AbstractDict &&
        haskey(solution_networks, "1")
    # TopologyOptimizer may report FEASIBLE_POINT after a callback-only run in which
    # Gurobi found no incumbent. A usable SCOPF solution must include the
    # primal network as well as a finite objective. This metric deliberately
    # permits positive overload slack; overload-free feasibility is reported
    # separately below.
    scopf_feasible = has_finite_objective &&
        string(get(result, "primal_status", "missing")) == "FEASIBLE_POINT" &&
        has_primal_network
    objective_components = get(data, "objective_costs", Dict())
    overload_cost_all = get(objective_components, "overload_all", nothing)
    overload_cost_base = get(objective_components, "overload_base_case", nothing)
    feasibility_tolerance = Float64(TopologyOptimizer.SOLVER_PRECISION)
    scopf_feasible_without_overload = scopf_feasible &&
        overload_cost_all isa Real && isfinite(overload_cost_all) &&
        abs(Float64(overload_cost_all)) <= feasibility_tolerance
    n_switching_actions = get(result, "n_switching_actions", 0)
    n_switching_actions <= topological_action_limit || error(
        "TAL=$topological_action_limit invariant violated: solver returned $n_switching_actions switching actions"
    )
    rating_validation = coupling_rating_validation(
        result, normalization, scopf_feasible, base_mva, network
    )

    output = Dict{String,Any}(
        "schema_version" => 5,
        "application" => APPLICATION,
        "formulation" => "DC",
        "algorithm" => "Benders",
        "termination_status" => string(get(result, "termination_status", "missing")),
        "primal_status" => string(get(result, "primal_status", "missing")),
        "has_finite_objective" => has_finite_objective,
        "has_primal_network" => has_primal_network,
        "scopf_feasible" => scopf_feasible,
        "scopf_feasibility_definition" => "certified finite incumbent with a primal network; overload slack permitted",
        "scopf_feasible_without_overload" => scopf_feasible_without_overload,
        "overload_cost_all_contingencies" => overload_cost_all,
        "overload_cost_base_case" => overload_cost_base,
        "overload_cost_feasibility_tolerance" => feasibility_tolerance,
        "objective_components" => json_safe(objective_components),
        "wall_seconds" => wall_seconds,
        "n_switching_actions" => n_switching_actions,
        "network" => Dict(
            "buses" => length(network["bus"]),
            "branches" => length(network["branch"]),
            "generators" => length(network["gen"]),
            "base_mva" => base_mva,
        ),
        "contingencies" => Dict(
            "level" => "N-1 branch failures",
            "single_branch_candidates" => single_branch_candidates,
            "retained_single_branch_failures" => retained_failures,
            "critical_or_disconnecting_failures_dropped" => max(
                0, single_branch_candidates - retained_failures
            ),
            "base_case_subproblem_entries" => base_case_entries,
            "policy" => "TopologyOptimizer excludes powered splitting and critical failures",
        ),
        "model_invariants" => Dict(
            "topological_action_limit" => TopologyOptimizer.PARAMS[:LIMITS][:BRANCH][:TAL_TIMESTEP],
            "switching_actions_allowed" => topological_action_limit > 0,
            "branch_switching_variable_layer" => TopologyOptimizer.PARAMS[:LIMITS][:BRANCH][:BRANCH_SWITCHING],
            "busbars" => false,
            "loss_estimation" => TopologyOptimizer.PARAMS[:LOSS_ESTIMATION][:ESTIMATE_LOSSES],
            "ac_feasibility_check" => TopologyOptimizer.PARAMS[:POSTPROCESSING][:AC_FEAS][:CHECK],
        ),
        "rating_normalization" => normalization,
        "rating_normalization_validation" => rating_validation,
        "time" => json_safe(get(result, "time", Dict())),
        "callback" => json_safe(get(result, "callback", Dict())),
        "info" => json_safe(get(result, "info", Dict())),
        "julia_version" => string(VERSION),
    )
    if has_finite_objective
        output["objective_value"] = Float64(objective_value)
    end
    package_version = try
        Base.pkgversion(TopologyOptimizer)
    catch
        nothing
    end
    package_version === nothing || (output["innoptem_version"] = string(package_version))

    open(result_file, "w") do stream
        JSON.print(stream, json_safe(output), 2)
        println(stream)
    end
    return nothing
end


try
    main(ARGS)
catch error_value
    showerror(stderr, error_value, catch_backtrace())
    println(stderr)
    exit(1)
end
