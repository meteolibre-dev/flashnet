#!/usr/bin/env bash
# slurm/submit_chain.sh
#
# RECOMMENDED ALTERNATIVE to relaunch_loop.sh.
#
# Submit N jobs as a Slurm dependency chain: each job starts automatically
# when the previous one ends. No need to keep any process alive on the login
# node (no tmux/nohup), which is much more reliable on Jean Zay.
#
#   afterok  = next job runs only if the previous one succeeded (chain stops on crash)
#   afterany = next job runs no matter how the previous one ended (always continue)
#
# For a 19h training job that you always want to restart (incl. when it hits
# the --time limit, i.e. state TIMEOUT), use afterany (the default).
#
# Usage:
#     bash slurm/submit_chain.sh                       # 10 runs, continue on any exit
#     N=5 DEPENDENCY=afterok bash slurm/submit_chain.sh
#
# Inspect:   squeue -u $USER
# Cancel all: scancel <first_jobid>   (cancelling the head cancels the chain)

set -uo pipefail

N="${N:-10}"
DEPENDENCY="${DEPENDENCY:-afterany}"
SLURM_SCRIPT="${SLURM_SCRIPT:-slurm/launch_jeanzay.slurm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"

prev=""
for i in $(seq 1 "$N"); do
    if [[ -z "$prev" ]]; then
        if ! jobid=$(sbatch --parsable "$SLURM_SCRIPT"); then
            echo "First submission failed (exit $?)." >&2
            exit 1
        fi
    else
        if ! jobid=$(sbatch --parsable --dependency="${DEPENDENCY}:${prev}" "$SLURM_SCRIPT"); then
            echo "Chain submission #$i failed (exit $?)." >&2
            exit 1
        fi
    fi
    echo "[$(date '+%F %T')] Run $i/$N -> job $jobid (depends on ${prev:-none})"
    prev="$jobid"
done

echo
echo "Submitted a chain of $N jobs. Check with:  squeue -u $USER"
echo "Cancel the whole chain with:               scancel <first_jobid>"
