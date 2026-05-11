# Source this file to pre-populate sbatch/srun flags:
#   source slurm_defaults.sh
# CLI flags still override these, so you can change one-off behavior easily.

# Refuse to run if executed (./slurm_defaults.sh) instead of sourced — the
# exports would be lost when the child shell exits.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: this file must be SOURCED, not executed." >&2
    echo "       Run:  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# --- sbatch defaults ---
export SBATCH_ACCOUNT=geohc
export SBATCH_TIMELIMIT=00:20:00
export SBATCH_QOS=high
# Note: there's no SBATCH_MEM env var; --mem must come from #SBATCH or CLI.

# --- srun / salloc defaults (also picked up by sbatch in many cases) ---
export SLURM_ACCOUNT=geohc
export SLURM_TIMELIMIT=00:20:00
export SLURM_QOS=high
export SLURM_MEM_PER_NODE=246064   # MB; up to 246064 normal, 2000000 bigmem

# --- chunk-worker (C/D) profile ---
# Derived from the partition the *current job* is running in (preferred) or
# from $SBATCH_PARTITION when sourced before submitting. bigmem nodes are
# whole-node allocations with 104 cores / 2 TB RAM, so we scale up cores and
# limit array concurrency to leave headroom for other users (only ~9 idle
# bigmem nodes exist cluster-wide). On standard partitions the historical
# 48 cores / 246 GB / 200-wide array still applies.
_chunk_part="${SLURM_JOB_PARTITION:-${SBATCH_PARTITION:-standard}}"
case "$_chunk_part" in
    bigmem*|medmem*)
        export CHUNK_PARTITION="$_chunk_part"
        export CHUNK_CPUS=104
        export CHUNK_MEM_MB=2000000
        export CHUNK_ARRAY_CONCURRENCY=6
        ;;
    *)
        export CHUNK_PARTITION="$_chunk_part"
        export CHUNK_CPUS=48
        export CHUNK_MEM_MB=246064
        export CHUNK_ARRAY_CONCURRENCY=200
        ;;
esac
unset _chunk_part
