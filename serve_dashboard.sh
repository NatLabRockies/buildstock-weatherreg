#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8787}"
SERVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/plots"
HOST="$(hostname -f)"
LOG="/tmp/dashboard_server_${PORT}.log"

listener_pid() { ss -ltnp 2>/dev/null | awk -F'pid=' "/:${PORT} /{split(\$2,a,\",\"); print a[1]}"; }

start() {
    if [[ -n "$(listener_pid)" ]]; then
        echo "Already running on port ${PORT} (pid $(listener_pid))."
        info; return 0
    fi
    setsid nohup python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$SERVE_DIR" \
        > "$LOG" 2>&1 < /dev/null &
    disown
    sleep 1
    if [[ -z "$(listener_pid)" ]]; then
        echo "Failed to start. Last log lines:"; tail -n 20 "$LOG"; exit 1
    fi
    echo "Started dashboard server (pid $(listener_pid)), serving ${SERVE_DIR}"
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

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    *) echo "Usage: $0 {start|stop|restart|status}"; exit 2 ;;
esac
