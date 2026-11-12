using Pkg

# Preserve the supplied dependency versions and avoid additional precompilation.
Pkg.instantiate(; allow_autoprecomp=false)

# Some pinned JLL packages require their lazy artifacts to exist before import.
for (uuid, package) in Pkg.dependencies()
    package.source === nothing && continue
    for filename in ("JuliaArtifacts.toml", "Artifacts.toml")
        path = joinpath(package.source, filename)
        isfile(path) || continue
        for (name, metadata) in Pkg.Artifacts.select_downloadable_artifacts(
            path; include_lazy=true, pkg_uuid=uuid,
        )
            Pkg.Artifacts.ensure_artifact_installed(name, metadata, path)
        end
        break
    end
end
