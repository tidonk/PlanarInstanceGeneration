#!/usr/bin/env bash
# Install pinned dependencies directly into a node-local depot.
pig_local_julia_depot() {
    local project="$1" destination="/local/tmp/$(id -un)/.julia-pig-installed"
    local filesystem key environment
    local installer="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/instantiate_julia.jl"
    mkdir -p "$destination"
    filesystem="$(findmnt -n -o FSTYPE -T "$destination")"
    case "$filesystem" in
        ext4|xfs|btrfs|zfs) ;;
        *) echo "Julia depot must be on local disk: $destination ($filesystem)" >&2; return 1 ;;
    esac
    export JULIA_DEPOT_PATH="$destination"
    export JULIA_PKG_PRECOMPILE_AUTO=0
    export JULIA_NUM_PRECOMPILE_TASKS=2
    export GUROBI_HOME="${GUROBI_HOME:-/opt/gurobi/linux64}"
    export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/opt/gurobi/gurobi.lic}"
    pig_local_julia_binary
    key="$(cat "$project/Project.toml" "$project/Manifest.toml" "$installer" | sha256sum | cut -d ' ' -f 1)"
    environment="$destination/environments/pig-$key"
    (
        flock -x 9 || exit
        if [[ ! -f "$environment/.instantiated" ]]; then
            mkdir -p "$environment" || exit
            # Only the two pinned environment files are staged, never package
            # caches. Keep Pkg from modifying the shared source environment.
            cp "$project/Project.toml" "$project/Manifest.toml" "$environment/" || exit
            ln -sfn "$project/src" "$environment/src" || exit
            printf 'Installing pinned Julia dependencies into %s\n' "$destination"
            timeout --kill-after=10s 900 "$PIG_JULIA" --startup-file=no --compiled-modules=existing --project="$environment" \
                "$installer" || exit
            cmp "$project/Manifest.toml" "$environment/Manifest.toml" || exit
            cmp "$project/Project.toml" "$environment/Project.toml" || exit
            touch "$environment/.instantiated" || exit
        fi
    ) 9>"$destination/.pig-install.lock" || return
    printf 'Julia depot: %s (%s)\n' "$JULIA_DEPOT_PATH" "$filesystem"
}

# Use the same Julia release as the original jobs, independently of Juliaup/NFS.
pig_local_julia_binary() {
    local root="/local/tmp/$(id -un)/julia-runtime"
    local archive="julia-1.12.6-linux-x86_64.tar.gz"
    mkdir -p "$root"
    (
        flock -x 9 || exit
        if [[ ! -f "$root/.verified-1.12.6" ]]; then
            curl --fail --location --max-time 300 \
                -o "$root/$archive" \
                "https://julialang-s3.julialang.org/bin/linux/x64/1.12/$archive" || exit
            # Published Julia 1.12.6 Linux x86_64 archive checksum.
            printf '%s  %s\n' bbabf3bef19421a9dbd24a767d807606ab85e444323b5a1c73ffe293fa3d079a \
                "$root/$archive" | sha256sum -c - || exit
            tar -xzf "$root/$archive" -C "$root" || exit
            "$root/julia-1.12.6/bin/julia" --startup-file=no --version || exit
            touch "$root/.verified-1.12.6" || exit
            rm "$root/$archive"
        fi
    ) 9>"$root/.install.lock" || return
    export PIG_JULIA="$root/julia-1.12.6/bin/julia"
}
