#!/usr/bin/env bash
set -euo pipefail

# Serve a built dashboard directory over HTTP so it's viewable via SSH tunnel.
# The build lives outside the code repo — pass the dashboard dir as the
# second arg, or set $DASHBOARD_DIR.
#
#   ./serve_dashboard.sh start /projects/geohc/radhikar/outputs/dashboard
#   DASHBOARD_DIR=/…/dashboard ./serve_dashboard.sh start
#
# Default location matches plots/I_build_dashboard.sh:
#   <parent(res_run_dir)>/dashboard/

PORT="${PORT:-8787}"
HOST="$(hostname -f)"
LOG="/tmp/dashboard_server_${PORT}.log"

cmd="${1:-start}"
serve_dir="${2:-${DASHBOARD_DIR:-}}"

listener_pid() { ss -ltnp 2>/dev/null | awk -F'pid=' "/:${PORT} /{split(\$2,a,\",\"); print a[1]}"; }

start() {
    if [[ -n "$(listener_pid)" ]]; then
        echo "Already running on port ${PORT} (pid $(listener_pid))."
        info; return 0
    fi
    if [[ -z "${serve_dir}" ]]; then
        echo "usage: $0 start <dashboard_dir>" >&2
        echo "       (or set \$DASHBOARD_DIR)" >&2
        exit 2
    fi
    if [[ ! -f "${serve_dir}/dashboard.html" ]]; then
        echo "ERROR: ${serve_dir}/dashboard.html not found." >&2
        echo "       Build first: sbatch plots/I_build_dashboard.sh <res_run_dir> <com_run_dir>" >&2
        exit 1
    fi
    setsid nohup python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$serve_dir" \
        > "$LOG" 2>&1 < /dev/null &
    disown
    sleep 1
    if [[ -z "$(listener_pid)" ]]; then
        echo "Failed to start. Last log lines:"; tail -n 20 "$LOG"; exit 1
    fi
    echo "Started dashboard server (pid $(listener_pid)), serving ${serve_dir}"
    info
}

stop() {
    local pid; pid="$(listener_pid)"
    if [[ -z "$pid" ]]; then echo "Nothing listening on port ${PORT}."; return 0; fi
    kill "$pid" && echo "Stopped server (pid ${pid})."
}

status() {
    local pid; pid="$(listener_pid)"
    if [[ -n "$pid" ]]; then echo "RUNNING on 127.0.0.1:${PORT} (pid ${pid})"; else echo "NOT running on port ${PORT}"; fi
}

info() {
    cat <<EOF

To reach it from your laptop, open an SSH tunnel (keep it running):
    ssh -N -L ${PORT}:localhost:${PORT} ${USER}@${HOST}
Then browse to:
    http://localhost:${PORT}/dashboard.html
EOF
}

case "${cmd}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    *) echo "Usage: $0 {start|stop|restart|status} [<dashboard_dir>]"; exit 2 ;;
esac
