#!/bin/bash
# Shared conda activation helper for all BLV runner scripts.
# Source this file; do not execute it directly.
#
# Supports:
#   - conda already on PATH (e.g. user shell init)
#   - CONDA_EXE hint (set by conda itself in child shells)
#   - common install locations (miniforge3, miniconda3, anaconda3)
#
# After sourcing, call:
#   activate_conda_env [ENV_NAME]   # default: openmmlab

_find_conda_sh() {
    if command -v conda &>/dev/null; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null)" || true
        if [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
            echo "${conda_base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    if [ -n "${CONDA_EXE:-}" ]; then
        local conda_dir
        conda_dir="$(dirname "$(dirname "${CONDA_EXE}")")"
        if [ -f "${conda_dir}/etc/profile.d/conda.sh" ]; then
            echo "${conda_dir}/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    local candidate
    for candidate in \
        "${HOME}/miniforge3" \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3" \
        "/opt/conda" \
        "/usr/local/conda" \
        "/modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1"; do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            echo "${candidate}/etc/profile.d/conda.sh"
            return 0
        fi
    done

    return 1
}

activate_conda_env() {
    local env_name="${1:-${CONDA_ENV_NAME:-openmmlab}}"

    if [ "${CONDA_DEFAULT_ENV:-}" = "${env_name}" ]; then
        return 0
    fi

    local conda_sh
    conda_sh="$(_find_conda_sh)" || {
        echo "ERROR: Could not locate conda installation." >&2
        echo "  Set CONDA_EXE or ensure conda is on PATH." >&2
        return 1
    }

    # shellcheck disable=SC1090
    source "${conda_sh}"
    # Some conda package activation scripts (e.g. activate-gcc_linux-64.sh)
    # reference variables like SYS_SYSROOT that may not be set.  With set -u
    # in the parent script this causes an "unbound variable" abort.  Temporarily
    # relax nounset around the activate call then restore it.
    local _restore_u=0
    if [[ $- == *u* ]]; then _restore_u=1; set +u; fi
    conda activate "${env_name}"
    if [[ ${_restore_u} -eq 1 ]]; then set -u; fi
}
