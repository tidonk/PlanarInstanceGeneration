#!/usr/bin/env julia
# Validate application feasibility independently of Benders cut generation.
using TopologyOptimizer
using JSON
using SHA
using TOML

function validate_compact(instance::String, settings::String, output::String)
    instance, settings, output = abspath.((instance, settings, output))
    application = "DC_SCOTS_TIME_SLACK"
    configuration = TOML.parsefile(settings)
    configuration["APPLICATION"] == application || error("compact application required")
    mkpath(output)
    directory_keys = (:SLACK_CACHE_DIR, :SLACKED_CASES_DIR, :SOL_DIR, :PLOT_DIR,
                      :INSTANCE_GEN_DIR, :SOLVER_LOG_DIR, :LOG_DIR, :RESULTS_DIR)
    redirects = Dict{Symbol,Any}(:USER_ENVIRONMENT => Dict{Symbol,Any}(
        :DIRECTORIES => Dict{Symbol,Any}(key => output for key in directory_keys)))
    # Reset cached logger streams, as in the existing application adapter.
    memento = TopologyOptimizer.Memento
    empty!(memento.gethandlers(TopologyOptimizer._LOGGER))
    push!(TopologyOptimizer._LOGGER, memento.DefaultHandler(stdout,
        memento.DefaultFormatter("{level}: {msg}"),
        Dict{Symbol,Any}(:is_colorized => false)))
    started = time()
    data, result = TopologyOptimizer.main(instance; application=application,
        slack_types=[:ratea], busbars=false, timesteps=1, plot=false,
        params_file=settings, additional_consts=redirects, return_data=true)
    network = data["nw"]["1"]
    objective = get(result, "objective", nothing)
    report = Dict(
        "instance" => instance,
        "instance_sha256" => bytes2hex(sha256(read(instance))),
        "settings" => configuration,
        "termination_status" => string(get(result, "termination_status", "missing")),
        "primal_status" => string(get(result, "primal_status", "missing")),
        "objective" => (objective isa Real && isfinite(objective) ? objective : nothing),
        "wall_seconds" => time() - started,
        "buses" => length(network["bus"]),
        "branches" => length(network["branch"]),
        "contingencies" => length(network["contingencies"]),
        "n_switching_actions" => get(result, "n_switching_actions", nothing),
    )
    open(joinpath(output, "validation.json"), "w") do stream
        JSON.print(stream, report, 2)
        println(stream)
    end
    println("COMPACT_VALIDATION ", JSON.json(report))
end

if abspath(PROGRAM_FILE) == @__FILE__
    !isempty(ARGS) && length(ARGS) % 3 == 0 || error(
        "usage: check_scopf_compact.jl INSTANCE SETTINGS OUTPUT_DIRECTORY [INSTANCE SETTINGS OUTPUT_DIRECTORY ...]")
    for offset in 1:3:length(ARGS)
        try
            validate_compact(ARGS[offset:offset+2]...)
        catch exception
            showerror(stderr, exception, catch_backtrace())
            println(stderr)
            output = abspath(ARGS[offset+2])
            mkpath(output)
            open(joinpath(output, "validation_error.txt"), "w") do stream
                showerror(stream, exception, catch_backtrace())
                println(stream)
            end
        end
        GC.gc()
    end
end
