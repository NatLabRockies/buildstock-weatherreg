# Source this from a launcher (e.g. A_start_building_stock_parallel_agg.sh)
# before invoking B. Sets the chunk-worker SLURM profile that B reads when
# sbatching the C array and F aggregate job:
#   CHUNK_PARTITION / CHUNK_CPUS / CHUNK_MEM_MB / CHUNK_ARRAY_CONCURRENCY
# The values track the partition the *launcher* is running in, so flipping
# A's `#SBATCH --partition=` cascades to the whole chain.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: ${BASH_SOURCE[0]} must be sourced, not executed." >&2
    exit 1
fi

_chunk_part="${SLURM_JOB_PARTITION:-${SBATCH_PARTITION:-standard}}"
export CHUNK_PARTITION="$_chunk_part"
case "$_chunk_part" in
    bigmem*|medmem*)
        # Whole-node alloc: 104 cores / 2 TB. Array capped so we leave headroom
        # for other users (only ~9 idle bigmem nodes cluster-wide).
        export CHUNK_CPUS=104
        export CHUNK_MEM_MB=2000000
        export CHUNK_ARRAY_CONCURRENCY=6
        ;;
    *)
        export CHUNK_CPUS=48
        export CHUNK_MEM_MB=246064
        export CHUNK_ARRAY_CONCURRENCY=200
        ;;
esac
unset _chunk_part
