#!/usr/bin/env bash
set -euo pipefail

# Starts or uses a persistent SSH ControlMaster socket for ORCD. The default
# socket path intentionally matches connect_remote.sh so diagnostics and the
# local Codex connector use the same remote login session.

REMOTE="${CODEX_ORCD_SSH:-${CODEX_REMOTE_DISCOVERY_SSH:-chang168@orcd-login.mit.edu}}"
CONTROL_DIR="${CODEX_REMOTE_SSH_CONTROL_DIR:-$HOME/.ssh/codex-remote-control}"
PERSIST="${CODEX_ORCD_CONTROL_PERSIST:-${CODEX_REMOTE_SSH_CONTROL_PERSIST:-12h}}"
RESET="${CODEX_ORCD_RESET_SSH:-${CODEX_REMOTE_RESET_SSH:-0}}"
OP_TIMEOUT="${CODEX_ORCD_CONTROL_OP_TIMEOUT:-${CODEX_REMOTE_SSH_CONTROL_OP_TIMEOUT:-8s}}"
SERVER_ALIVE_INTERVAL="${CODEX_ORCD_SERVER_ALIVE_INTERVAL:-${CODEX_REMOTE_SSH_SERVER_ALIVE_INTERVAL:-30}}"
SERVER_ALIVE_COUNT_MAX="${CODEX_ORCD_SERVER_ALIVE_COUNT_MAX:-${CODEX_REMOTE_SSH_SERVER_ALIVE_COUNT_MAX:-3}}"
CONNECTION_ATTEMPTS="${CODEX_ORCD_CONNECTION_ATTEMPTS:-${CODEX_REMOTE_SSH_CONNECTION_ATTEMPTS:-1}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */scripts/remote/codex ]]; then
    DEFAULT_LOCAL_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
else
    DEFAULT_LOCAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi
CODEX_REMOTE_PROJECT_NAME="${CODEX_REMOTE_PROJECT_NAME:-$(basename "$DEFAULT_LOCAL_ROOT")}"

sanitize_key() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9_.@=-' '_'
}

SOCKET="${CODEX_ORCD_SOCKET:-$CONTROL_DIR/$(sanitize_key "$REMOTE")}"

mkdir -p "$(dirname "$SOCKET")"
chmod 700 "$(dirname "$SOCKET")" 2>/dev/null || true

usage() {
    cat >&2 <<EOF
Usage: $0 {start|status|diagnose|run|stop|path} [remote command...]

Starts or uses a persistent SSH ControlMaster socket for ORCD.

Environment:
  CODEX_ORCD_SSH              default: chang168@orcd-login.mit.edu
  CODEX_ORCD_SOCKET           default: ~/.ssh/codex-remote-control/<remote>
  CODEX_ORCD_CONTROL_PERSIST  default: 12h
  CODEX_REMOTE_RESET_SSH      remove any existing socket before starting
  CODEX_REMOTE_PROJECT_NAME   default: basename of this local checkout

Examples:
  $0 start
  $0 diagnose
  $0 run 'hostname -f'
EOF
}

run_with_timeout() {
    local duration="$1"
    shift

    if command -v timeout >/dev/null 2>&1; then
        timeout "$duration" "$@"
    else
        "$@"
    fi
}

check_socket() {
    [[ -S "$SOCKET" ]] || {
        echo "ORCD SSH control socket is not present: $SOCKET" >&2
        return 1
    }
    run_with_timeout "$OP_TIMEOUT" ssh -S "$SOCKET" -o BatchMode=yes -O check "$REMOTE"
}

start_socket() {
    if [[ "$RESET" == "1" ]]; then
        if [[ -S "$SOCKET" ]]; then
            ssh -S "$SOCKET" -o BatchMode=yes -O exit "$REMOTE" >/dev/null 2>&1 || true
        fi
        rm -f -- "$SOCKET"
    fi

    if [[ -S "$SOCKET" ]] && check_socket >/dev/null 2>&1; then
        echo "ORCD SSH control socket is already running: $SOCKET"
        exit 0
    fi

    if [[ -e "$SOCKET" ]]; then
        echo "Removing stale ORCD SSH control socket: $SOCKET" >&2
        rm -f -- "$SOCKET"
    fi

    ssh -MNf \
        -S "$SOCKET" \
        -o ControlMaster=yes \
        -o ControlPersist="$PERSIST" \
        -o "ServerAliveInterval=${SERVER_ALIVE_INTERVAL}" \
        -o "ServerAliveCountMax=${SERVER_ALIVE_COUNT_MAX}" \
        -o "ConnectionAttempts=${CONNECTION_ATTEMPTS}" \
        "$REMOTE"

    check_socket
    echo "ORCD SSH control socket is ready: $SOCKET"
}

run_remote() {
    check_socket >/dev/null
    ssh -S "$SOCKET" -o BatchMode=yes "$REMOTE" "$@"
}

diagnose_remote() {
    check_socket >/dev/null
    ssh -S "$SOCKET" -o BatchMode=yes "$REMOTE" 'bash -s' -- "$CODEX_REMOTE_PROJECT_NAME" <<'REMOTE_DIAG'
set -u

project_name="$1"
state_dirs=(
    "/orcd/scratch/orcd/011/chang168/${project_name}/scripts/remote/codex"
    "/orcd/scratch/orcd/011/chang168/${project_name}/codex"
    "/home/chang168/orcd/scratch/${project_name}/scripts/remote/codex"
    "/home/chang168/orcd/scratch/${project_name}/codex"
    "/orcd/pool/006/chang168/${project_name}/scripts/remote/codex"
    "/orcd/pool/006/chang168/${project_name}/codex"
    "/home/chang168/orcd/pool/${project_name}/scripts/remote/codex"
    "/home/chang168/orcd/pool/${project_name}/codex"
    "~/orcd/scratch/${project_name}/scripts/remote/codex"
    "~/orcd/scratch/${project_name}/codex"
    "~/orcd/pool/${project_name}/scripts/remote/codex"
    "~/orcd/pool/${project_name}/codex"
    "~/${project_name}/scripts/remote/codex"
    "~/${project_name}/codex"
)

current_host="$(hostname -f 2>/dev/null || hostname)"

host_matches_current() {
    local host="$1"
    local short="${host%%.*}"

    [[ -z "$host" || "$host" == "127.0.0.1" || "$host" == "localhost" ]] && return 0
    [[ "$host" == "$current_host" || "$short" == "${current_host%%.*}" ]]
}

readyz() {
    local bind_host="$1"
    local host="$2"
    local port="$3"
    local target
    local status_line

    [[ "$port" =~ ^[0-9]+$ ]] || return 2
    if [[ -z "$bind_host" || "$bind_host" == "127.0.0.1" || "$bind_host" == "localhost" ]]; then
        if ! host_matches_current "$host"; then
            return 3
        fi
        target="127.0.0.1"
    else
        target="$bind_host"
    fi

    if command -v curl >/dev/null 2>&1; then
        curl --noproxy '*' --fail --silent --show-error --max-time 2 \
            "http://${target}:${port}/readyz" >/dev/null
        return
    fi

    exec 3<>"/dev/tcp/${target}/${port}" || return 1
    printf 'GET /readyz HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n' >&3
    IFS= read -r status_line <&3 || status_line=""
    exec 3>&-
    exec 3<&-
    [[ "$status_line" == *" 200 "* ]]
}

echo "== ORCD SSH =="
echo "host: $current_host"
echo "project: $project_name"
echo "time: $(date -Is 2>/dev/null || date)"
echo

echo "== Codex CLI =="
if command -v codex >/dev/null 2>&1; then
    echo "codex: $(command -v codex)"
    codex --version 2>/dev/null || true
else
    echo "codex: not found on this login shell PATH"
fi
echo

echo "== App Server Processes =="
found_process=0
while read -r pid args; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "$args" == *"codex app-server"* ]] || continue
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    if [[ -n "$project_name" && -n "$cwd" ]] &&
        [[ "$(basename "$cwd")" != "$project_name" ]] &&
        [[ "$cwd" != *"/${project_name}/"* ]]; then
        continue
    fi
    found_process=1
    printf '%s %s cwd=%s\n' "$pid" "$args" "${cwd:-unknown}"
done < <(ps -u "$(id -u)" -o pid= -o args= 2>/dev/null)
[[ "$found_process" == "1" ]] || echo "no ${project_name} codex app-server process found on $current_host"
echo

echo "== State Files =="
found_state=0
shopt -s nullglob
for dir in "${state_dirs[@]}"; do
    expanded="${dir/#\~/$HOME}"
    files=("${expanded}"/remote_server.*.env "${expanded}/remote_server.env")
    (( ${#files[@]} > 0 )) || continue
    mapfile -t files < <(ls -t -- "${files[@]}" 2>/dev/null)
    for state_file in "${files[@]}"; do
        [[ -r "$state_file" ]] || continue
        found_state=1
        unset CODEX_REMOTE_SERVER_HOST CODEX_REMOTE_SERVER_BIND_HOST CODEX_REMOTE_SERVER_PORT
        unset CODEX_REMOTE_SERVER_WORKDIR CODEX_REMOTE_SERVER_NAME CODEX_REMOTE_SERVER_PID
        unset CODEX_REMOTE_SERVER_CODEX_HOME CODEX_REMOTE_SERVER_STARTED_AT CODEX_REMOTE_SERVER_SLURM_JOB_ID
        # shellcheck disable=SC1090
        source "$state_file" || {
            echo "-- $state_file"
            echo "status: could not source"
            continue
        }
        echo "-- $state_file"
        echo "name: ${CODEX_REMOTE_SERVER_NAME:-default}"
        echo "host: ${CODEX_REMOTE_SERVER_HOST:-unknown}"
        echo "bind: ${CODEX_REMOTE_SERVER_BIND_HOST:-unknown}"
        echo "port: ${CODEX_REMOTE_SERVER_PORT:-unknown}"
        echo "workdir: ${CODEX_REMOTE_SERVER_WORKDIR:-unknown}"
        echo "pid: ${CODEX_REMOTE_SERVER_PID:-unknown}"
        echo "codex_home: ${CODEX_REMOTE_SERVER_CODEX_HOME:-unknown}"
        echo "started_at: ${CODEX_REMOTE_SERVER_STARTED_AT:-unknown}"
        if [[ "${CODEX_REMOTE_SERVER_PID:-}" =~ ^[0-9]+$ ]] &&
            kill -0 "$CODEX_REMOTE_SERVER_PID" >/dev/null 2>&1; then
            pid_alive=1
            echo "pid_alive: yes"
        else
            pid_alive=0
            echo "pid_alive: no"
        fi
        if [[ "${CODEX_REMOTE_SERVER_PID:-}" =~ ^[0-9]+$ ]] &&
            [[ "$pid_alive" != "1" ]] &&
            host_matches_current "${CODEX_REMOTE_SERVER_HOST:-}"; then
            echo "readyz: skipped; state PID is not alive on this host"
        elif readyz \
            "${CODEX_REMOTE_SERVER_BIND_HOST:-}" \
            "${CODEX_REMOTE_SERVER_HOST:-}" \
            "${CODEX_REMOTE_SERVER_PORT:-}"; then
            echo "readyz: ok"
        else
            ready_status=$?
            case "$ready_status" in
                3)
                    echo "readyz: skipped; state is bound to localhost on another host"
                    ;;
                *)
                    echo "readyz: failed"
                    ;;
            esac
        fi
        echo
    done
done
shopt -u nullglob

if [[ "$found_state" != "1" ]]; then
    echo "no readable remote_server*.env files found in known ORCD paths"
fi
REMOTE_DIAG
}

case "${1:-start}" in
    start)
        start_socket
        ;;
    status)
        check_socket
        ;;
    diagnose)
        diagnose_remote
        ;;
    run)
        shift
        if [[ "$#" -eq 0 ]]; then
            usage
            exit 2
        fi
        run_remote "$@"
        ;;
    stop)
        check_socket >/dev/null
        ssh -S "$SOCKET" -o BatchMode=yes -O exit "$REMOTE"
        ;;
    path)
        printf '%s\n' "$SOCKET"
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
