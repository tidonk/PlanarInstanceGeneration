#!/usr/bin/env julia

using BendersNetworkDesign
using Gurobi
using JuMP
using TOML


# Upstream build_master_problem calls read_settings() without forwarding the
# Settings object passed to solve_benders. Install the process-local override
# before main is defined so Julia's world-age rules cannot hide it from the
# solve call. The external checkout is never modified.
@eval BendersNetworkDesign begin
    const PLANAR_INSTANCE_GENERATION_RUNNER_SETTINGS =
        Ref{Union{Nothing,Settings}}(nothing)

    function read_settings()::Settings
        settings = PLANAR_INSTANCE_GENERATION_RUNNER_SETTINGS[]
        settings === nothing && error(
            "planar-instance-generation runner settings are not initialized"
        )
        return settings
    end
end


function optional_finite(callable)
    try
        value = callable()
        return isfinite(value) && abs(value) < 1.0e90 ?
            Float64(value) : nothing
    catch
        return nothing
    end
end


function main(args)::Nothing
    length(args) == 4 || error(
        "usage: run_snd_benders.jl INSTANCE SETTINGS RESULT THREADS"
    )
    instance_file, settings_file, result_file, threads_text = args
    threads = parse(Int, threads_text)
    threads >= 1 || error("THREADS must be positive")

    settings = read_settings(settings_file)
    settings.model_type == "benders" || error("runner requires model_type=benders")
    settings.contingency_k == 1 || error("runner is restricted to N-1 contingencies")
    settings.num_outage_scenarios == -1 || error("runner must enumerate all contingencies")
    BendersNetworkDesign.PLANAR_INSTANCE_GENERATION_RUNNER_SETTINGS[] = settings
    active_settings = BendersNetworkDesign.read_settings()
    active_settings === settings || error("process-local settings override failed")
    active_settings.time_limit == settings.time_limit || error(
        "process-local settings time limit does not match the requested value"
    )

    network = read_sndlib_network(instance_file)
    restricted_demands = sort([
        demand.id for demand in values(network.demands)
        if demand.max_path_length !== nothing || !isempty(demand.admissible_paths)
    ])
    isempty(restricted_demands) || error(
        "BendersNetworkDesign does not enforce maxPathLength/admissiblePaths; " *
        "refusing restricted demand(s): " * join(restricted_demands, ", ")
    )
    scenario_started = time_ns()
    scenarios = generate_outage_scenarios(
        network; include_base_case=false, k=settings.contingency_k
    )
    scenario_seconds = (time_ns() - scenario_started) / 1.0e9
    number_of_links = length(network.network_structure.links)
    candidate_count = binomial(number_of_links, settings.contingency_k)

    optimizer = JuMP.optimizer_with_attributes(
        Gurobi.Optimizer,
        "Threads" => threads,
        "OutputFlag" => 1,
        "TimeLimit" => settings.time_limit,
    )
    solve_started = time_ns()
    result = solve_benders(
        network;
        optimizer=optimizer,
        outage_scenarios=scenarios,
        settings=settings,
    )
    wall_seconds = (time_ns() - solve_started) / 1.0e9
    effective_time_limit = JuMP.time_limit_sec(result.model)
    effective_time_limit === nothing && error(
        "master model does not expose an effective time limit"
    )
    isapprox(
        effective_time_limit,
        settings.time_limit;
        rtol=0.0,
        atol=1.0e-9,
    ) || error(
        "master time limit $(effective_time_limit) does not match requested " *
        "$(settings.time_limit)"
    )

    output = Dict{String,Any}(
        "termination_status" => string(result.status),
        "has_finite_objective" => isfinite(result.objective_value),
        "iterations" => result.iterations,
        "installed_module_variables" => length(result.y_solution),
        "total_cuts_added" => result.total_cuts_added,
        "total_cuts_found" => result.total_cuts_found,
        "total_subproblems_solved" => result.total_subproblems_solved,
        "solver_seconds" => result.total_solve_time,
        "wall_seconds" => wall_seconds,
        "callback_seconds" => result.total_callback_time,
        "subproblem_seconds" => result.total_subproblem_solve_time,
        "master_seconds" => result.total_master_time,
        "node_count" => result.node_count,
        "scenario_generation_seconds" => scenario_seconds,
        "network_nodes" => length(network.network_structure.nodes),
        "network_links" => number_of_links,
        "network_demands" => length(network.demands),
        "routing_model" => "unrestricted point-to-point demands",
        "routing_restrictions_present" => false,
        "outage_candidates" => candidate_count,
        "outage_scenarios_retained" => length(scenarios),
        "disconnecting_outages_dropped" => candidate_count - length(scenarios),
        "base_case_in_master" => true,
        "base_case_as_subproblem" => false,
        "contingency_k" => settings.contingency_k,
        "time_limit_seconds" => settings.time_limit,
        "requested_time_limit_seconds" => settings.time_limit,
        "effective_time_limit_seconds" => effective_time_limit,
        "time_limit_verified" => true,
        "threads" => threads,
        "julia_version" => string(VERSION),
    )
    if isfinite(result.objective_value)
        output["objective_value"] = result.objective_value
    end
    objective_bound_value = optional_finite(() -> JuMP.objective_bound(result.model))
    relative_gap_value = optional_finite(() -> JuMP.relative_gap(result.model))
    objective_bound_value === nothing || (output["objective_bound"] = objective_bound_value)
    relative_gap_value === nothing || (output["relative_gap"] = relative_gap_value)
    package_version = try
        Base.pkgversion(BendersNetworkDesign)
    catch
        nothing
    end
    package_version === nothing || (
        output["benders_network_design_version"] = string(package_version)
    )

    mkpath(dirname(result_file))
    open(result_file * ".tmp", "w") do stream
        TOML.print(stream, output; sorted=true)
    end
    mv(result_file * ".tmp", result_file; force=true)
    return nothing
end


try
    main(ARGS)
catch error_value
    showerror(stderr, error_value, catch_backtrace())
    println(stderr)
    exit(1)
end
