#!/usr/bin/env bash
set -euo pipefail

# Connect the local Codex TUI to one live remote app-server.
#
# The default selector is "available": choose the newest live server that is
# not already claimed by another connector on this local machine. Explicit
# server ids, names, and ports are also supported.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Run this script instead of sourcing it: bash codex/connect_remote.sh" >&2
    return 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */scripts/remote/codex ]]; then
    DEFAULT_LOCAL_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
else
    DEFAULT_LOCAL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fi

PROJECT_NAME="${CODEX_REMOTE_PROJECT_NAME:-$(basename "$DEFAULT_LOCAL_ROOT")}"
DISCOVERY_SSH="${CODEX_REMOTE_DISCOVERY_SSH:-${CODEX_REMOTE_LOGIN:-chang168@orcd-login.mit.edu}}"
REMOTE_ROOT="${CODEX_REMOTE_ROOT:-}"
SERVER_SELECTOR="${CODEX_REMOTE_SERVER_ID:-${1:-available}}"
LOCAL_PORT_REQUESTED="${CODEX_LOCAL_PORT:-}"
LOCAL_PORT_SCAN_LIMIT="${CODEX_LOCAL_PORT_SCAN_LIMIT:-100}"
ALLOW_REUSE="${CODEX_REMOTE_ALLOW_REUSE:-0}"
SANDBOX="${CODEX_REMOTE_SANDBOX:-danger-full-access}"
APPROVAL="${CODEX_REMOTE_APPROVAL:-on-request}"
MODEL="${CODEX_MODEL:-}"
SESSION_MODE="${CODEX_REMOTE_SESSION_MODE:-new}"
SESSION_ID="${CODEX_REMOTE_SESSION_ID:-}"
RESUME_ALL="${CODEX_REMOTE_RESUME_ALL:-0}"
READY_TIMEOUT="${CODEX_REMOTE_READY_TIMEOUT:-10}"

SSH_CONTROL_DIR="${CODEX_REMOTE_SSH_CONTROL_DIR:-$HOME/.ssh/codex-remote-control}"
SSH_CONTROL_PERSIST="${CODEX_REMOTE_SSH_CONTROL_PERSIST:-8h}"
SSH_OP_TIMEOUT="${CODEX_REMOTE_SSH_OP_TIMEOUT:-30s}"
SSH_HEALTH_TIMEOUT="${CODEX_REMOTE_SSH_HEALTH_TIMEOUT:-12s}"
SSH_OPEN_TIMEOUT="${CODEX_REMOTE_SSH_OPEN_TIMEOUT:-300s}"
RESET_SSH="${CODEX_REMOTE_RESET_SSH:-0}"
CLAIM_ROOT="${CODEX_REMOTE_CLAIM_DIR:-${XDG_RUNTIME_DIR:-/tmp}/codex-remote-claims-$(id -u)/$PROJECT_NAME}"

CONTROL_PATH=""
TUNNEL_CONTROL_PATH=""
TUNNEL_REMOTE=""
TUNNEL_STARTED=0
FORWARD_SPEC=""
LOCAL_PORT=""
CLAIM_FILE=""
CLAIM_HELD=0
SERVERS_TMP=""

CODEX_REMOTE_SERVER_ID=""
CODEX_REMOTE_SERVER_NAME=""
CODEX_REMOTE_SERVER_HOST=""
CODEX_REMOTE_SERVER_BIND_HOST=""
CODEX_REMOTE_SERVER_PORT=""
CODEX_REMOTE_SERVER_WORKDIR=""
CODEX_REMOTE_SERVER_CODEX_HOME=""
CODEX_REMOTE_SERVER_SQLITE_HOME=""
CODEX_REMOTE_SERVER_TOKEN_FILE=""
CODEX_REMOTE_DISCOVERY_HOST=""
CODEX_REMOTE_APP_TOKEN=""

info() {
    printf '%s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

sanitize_key() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9_.@=-' '_'
}

short_key() {
    local prefix="$1"
    local value="$2"
    local hash
    if command -v cksum >/dev/null 2>&1; then
        hash="$(printf '%s' "$value" | cksum | awk '{print $1}')"
    else
        hash="$(sanitize_key "$value" | cut -c 1-40)"
    fi
    printf '%s_%s' "$prefix" "$hash"
}

run_timeout() {
    local duration="$1"
    shift
    if command -v timeout >/dev/null 2>&1; then
        if timeout --help 2>/dev/null | grep -q -- '--foreground'; then
            timeout --foreground "$duration" "$@"
        else
            timeout "$duration" "$@"
        fi
    else
        "$@"
    fi
}

valid_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

same_host() {
    local first="$1"
    local second="$2"
    [[ -n "$first" && -n "$second" ]] || return 1
    [[ "$first" == "$second" || "${first%%.*}" == "${second%%.*}" ]]
}

local_port_is_listening() {
    local port="$1"
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

select_local_port() {
    local preferred="$1"
    local candidate
    local last_port

    valid_int "$preferred" || die "remote port is invalid: $preferred"
    if [[ -n "$LOCAL_PORT_REQUESTED" ]]; then
        valid_int "$LOCAL_PORT_REQUESTED" || die "CODEX_LOCAL_PORT must be an integer"
        local_port_is_listening "$LOCAL_PORT_REQUESTED" && die "local port $LOCAL_PORT_REQUESTED is already in use"
        LOCAL_PORT="$LOCAL_PORT_REQUESTED"
        return
    fi

    valid_int "$LOCAL_PORT_SCAN_LIMIT" || die "CODEX_LOCAL_PORT_SCAN_LIMIT must be an integer"
    (( LOCAL_PORT_SCAN_LIMIT > 0 )) || die "CODEX_LOCAL_PORT_SCAN_LIMIT must be positive"
    last_port=$(( preferred + LOCAL_PORT_SCAN_LIMIT - 1 ))
    (( last_port > 65535 )) && last_port=65535
    for ((candidate = preferred; candidate <= last_port; candidate++)); do
        if ! local_port_is_listening "$candidate"; then
            LOCAL_PORT="$candidate"
            return
        fi
    done
    die "no free local port found in ${preferred}-${last_port}"
}

control_master_is_running() {
    [[ -S "$CONTROL_PATH" ]] || return 1
    run_timeout "$SSH_OP_TIMEOUT" ssh -S "$CONTROL_PATH" -o BatchMode=yes -O check "$DISCOVERY_SSH" >/dev/null 2>&1 || return 1
}

control_master_responds() {
    control_master_is_running || return 1
    run_timeout "$SSH_HEALTH_TIMEOUT" ssh -S "$CONTROL_PATH" -o BatchMode=yes "$DISCOVERY_SSH" true >/dev/null 2>&1
}

close_control_master() {
    if [[ -S "$CONTROL_PATH" ]]; then
        run_timeout 5s ssh -S "$CONTROL_PATH" -O exit "$DISCOVERY_SSH" >/dev/null 2>&1 || true
    fi
    rm -f -- "$CONTROL_PATH"
}

open_control_master() {
    mkdir -p "$SSH_CONTROL_DIR"
    chmod 700 "$SSH_CONTROL_DIR" 2>/dev/null || true
    CONTROL_PATH="$SSH_CONTROL_DIR/$(sanitize_key "$DISCOVERY_SSH")"

    if [[ "$RESET_SSH" == "1" ]]; then
        close_control_master
    fi

    if control_master_responds; then
        info "Reusing SSH control connection: $DISCOVERY_SSH"
        return
    fi

    if [[ -S "$CONTROL_PATH" ]]; then
        info "Discarding unusable SSH control connection: $DISCOVERY_SSH"
        close_control_master
    fi
    info "Opening SSH control connection: $DISCOVERY_SSH"
    info "Complete the ORCD password and Duo prompt if SSH asks for it."
    info "ORCD will briefly open a login shell, then the connector will close it and keep the authenticated control channel."
    if ! run_timeout "$SSH_OPEN_TIMEOUT" ssh \
        -tt \
        -S "$CONTROL_PATH" \
        -o ControlMaster=yes \
        -o "ControlPersist=$SSH_CONTROL_PERSIST" \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ConnectionAttempts=1 \
        "$DISCOVERY_SSH" <<< 'exit'; then
        close_control_master
        die "the initial ORCD login shell did not complete; rerun the connector and finish the password/Duo prompt"
    fi
    if ! control_master_is_running; then
        close_control_master
        die "SSH authenticated, but the persistent control connection did not remain available"
    fi
}

ssh_discovery() {
    run_timeout "$SSH_OP_TIMEOUT" ssh -S "$CONTROL_PATH" -o BatchMode=yes "$DISCOVERY_SSH" "$@"
}

read_remote_servers() {
    local output="$1"
    ssh_discovery 'bash -s' -- "$PROJECT_NAME" "$REMOTE_ROOT" > "$output" <<'REMOTE'
set -euo pipefail
project="$1"
root_override="${2:-}"
root="${root_override:-$HOME/.codex-remote/$project}"
servers="$root/servers"
ttl="${CODEX_REMOTE_STALE_SECONDS:-120}"
now="$(date +%s)"
discovery_host="$(hostname -f 2>/dev/null || hostname)"

shopt -s nullglob
for lease in "$servers"/*.env; do
    unset CODEX_REMOTE_SERVER_ID CODEX_REMOTE_SERVER_NAME CODEX_REMOTE_SERVER_STATUS
    unset CODEX_REMOTE_SERVER_HOST CODEX_REMOTE_SERVER_BIND_HOST CODEX_REMOTE_SERVER_PORT
    unset CODEX_REMOTE_SERVER_WORKDIR CODEX_REMOTE_SERVER_CODEX_HOME CODEX_REMOTE_SERVER_SQLITE_HOME
    unset CODEX_REMOTE_SERVER_TOKEN_FILE
    unset CODEX_REMOTE_SERVER_HEARTBEAT CODEX_REMOTE_SERVER_STARTED_AT
    # shellcheck disable=SC1090
    source "$lease" || continue
    heartbeat="${CODEX_REMOTE_SERVER_HEARTBEAT:-}"
    fresh="stale"
    mtime=0
    if [[ -n "$heartbeat" && -e "$heartbeat" ]]; then
        mtime="$(stat -c '%Y' "$heartbeat" 2>/dev/null || printf '0')"
        age="$(( now - mtime ))"
        if (( age <= ttl )) && [[ "${CODEX_REMOTE_SERVER_STATUS:-}" == "live" ]]; then
            fresh="live"
        fi
    fi
    printf '%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\037%s\n' \
        "$mtime" \
        "${CODEX_REMOTE_SERVER_ID:-unknown}" \
        "${CODEX_REMOTE_SERVER_NAME:-default}" \
        "${CODEX_REMOTE_SERVER_STATUS:-unknown}" \
        "$fresh" \
        "${CODEX_REMOTE_SERVER_HOST:-unknown}" \
        "${CODEX_REMOTE_SERVER_BIND_HOST:-127.0.0.1}" \
        "${CODEX_REMOTE_SERVER_PORT:-0}" \
        "${CODEX_REMOTE_SERVER_WORKDIR:-}" \
        "${CODEX_REMOTE_SERVER_CODEX_HOME:-}" \
        "${CODEX_REMOTE_SERVER_SQLITE_HOME:-}" \
        "${CODEX_REMOTE_SERVER_TOKEN_FILE:-}" \
        "$discovery_host" \
        "$lease" \
        "${CODEX_REMOTE_SERVER_STARTED_AT:-unknown}"
done | sort -t $'\037' -k15,15r
REMOTE
}

print_servers() {
    local show_all="${1:-0}"
    local _mtime id name status freshness host _bind port workdir shared sqlite _token _discovery _lease started
    printf 'SERVER_ID\tNAME\tSTATE\tHOST:PORT\tSTARTED\tWORKDIR\n'
    while IFS=$'\037' read -r _mtime id name status freshness host _bind port workdir shared sqlite _token _discovery _lease started; do
        [[ -n "$id" ]] || continue
        if [[ "$show_all" != "1" ]] && { [[ "$status" != "live" ]] || [[ "$freshness" != "live" ]]; }; then
            continue
        fi
        printf '%s\t%s\t%s/%s\t%s:%s\t%s\t%s\n' \
            "$id" "$name" "$status" "$freshness" "$host" "$port" "$started" "$workdir"
    done < "$SERVERS_TMP"
}

try_claim() {
    local id="$1"
    local key
    if [[ "$ALLOW_REUSE" == "1" ]] || ! command -v flock >/dev/null 2>&1; then
        return 0
    fi

    mkdir -p "$CLAIM_ROOT"
    chmod 700 "$(dirname "$CLAIM_ROOT")" "$CLAIM_ROOT" 2>/dev/null || true
    key="$(short_key server "$id")"
    CLAIM_FILE="$CLAIM_ROOT/$key.lock"
    exec 9>"$CLAIM_FILE"
    if flock -n 9; then
        CLAIM_HELD=1
        printf 'pid=%s\nserver=%s\nstarted_at=%s\n' "$$" "$id" "$(date -Is)" >&9
        return 0
    fi
    exec 9>&-
    CLAIM_FILE=""
    return 1
}

select_server() {
    local _mtime id name status freshness host bind port workdir shared sqlite token discovery lease _started
    local matched=0
    local claimed=0

    while IFS=$'\037' read -r _mtime id name status freshness host bind port workdir shared sqlite token discovery lease _started; do
        [[ -n "$id" ]] || continue
        case "$SERVER_SELECTOR" in
            available|latest)
                ;;
            *)
                if [[ "$id" != "$SERVER_SELECTOR" && "$name" != "$SERVER_SELECTOR" && "$port" != "$SERVER_SELECTOR" ]]; then
                    continue
                fi
                matched=1
                ;;
        esac

        if [[ "$status" != "live" || "$freshness" != "live" ]]; then
            continue
        fi

        if [[ "$SERVER_SELECTOR" == "latest" ]]; then
            ALLOW_REUSE=1
        fi
        if ! try_claim "$id"; then
            claimed=$((claimed + 1))
            continue
        fi

        CODEX_REMOTE_SERVER_ID="$id"
        CODEX_REMOTE_SERVER_NAME="$name"
        CODEX_REMOTE_SERVER_HOST="$host"
        CODEX_REMOTE_SERVER_BIND_HOST="$bind"
        CODEX_REMOTE_SERVER_PORT="$port"
        CODEX_REMOTE_SERVER_WORKDIR="$workdir"
        CODEX_REMOTE_SERVER_CODEX_HOME="$shared"
        CODEX_REMOTE_SERVER_SQLITE_HOME="$sqlite"
        CODEX_REMOTE_SERVER_TOKEN_FILE="$token"
        CODEX_REMOTE_DISCOVERY_HOST="$discovery"
        return 0
    done < "$SERVERS_TMP"

    if [[ "$matched" == "1" ]]; then
        die "selected server is not live or is already claimed: $SERVER_SELECTOR"
    fi
    if (( claimed > 0 )); then
        die "all live Codex servers are already connected locally; start another remote server or set CODEX_REMOTE_ALLOW_REUSE=1"
    fi
    die "no live Codex server matched '$SERVER_SELECTOR'; run '$0 list'"
}

open_tunnel() {
    local server_host="$1"
    local discovery_host="$2"
    local bind_host="$3"
    local remote_port="$4"
    local target

    case "$bind_host" in
        ""|127.0.0.1|localhost)
            if ! same_host "$server_host" "$discovery_host"; then
                die "server $server_host is loopback-bound and the SSH control connection is on $discovery_host; restart it with the updated start_remote_server.sh"
            fi
            target="127.0.0.1"
            ;;
        *)
            target="$bind_host"
            ;;
    esac

    FORWARD_SPEC="${LOCAL_PORT}:${target}:${remote_port}"
    info "Opening SSH tunnel through ${DISCOVERY_SSH}: localhost:${LOCAL_PORT} -> ${target}:${remote_port} (${server_host})"
    ssh -S "$CONTROL_PATH" \
        -O forward \
        -o ExitOnForwardFailure=yes \
        -L "$FORWARD_SPEC" \
        "$DISCOVERY_SSH"
    TUNNEL_REMOTE="$DISCOVERY_SSH"
    TUNNEL_STARTED=1
}

load_remote_token() {
    local token_file="$1"
    [[ -n "$token_file" ]] || return 0
    CODEX_REMOTE_APP_TOKEN="$(ssh_discovery 'bash -s' -- "$token_file" <<'REMOTE'
set -euo pipefail
token_file="$1"
[[ -f "$token_file" && -r "$token_file" ]] || exit 1
cat "$token_file"
REMOTE
)" || die "could not read the remote app-server token"
    [[ -n "$CODEX_REMOTE_APP_TOKEN" ]] || die "remote app-server token is empty"
    export CODEX_REMOTE_APP_TOKEN
}

websocket_auth_is_available() {
    local status_line
    [[ -n "$CODEX_REMOTE_APP_TOKEN" ]] || return 0
    exec 3<>"/dev/tcp/127.0.0.1/${LOCAL_PORT}" || return 1
    printf 'GET / HTTP/1.1\r\nHost: 127.0.0.1:%s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nAuthorization: Bearer %s\r\n\r\n' \
        "$LOCAL_PORT" "$CODEX_REMOTE_APP_TOKEN" >&3
    IFS= read -r status_line <&3 || status_line=""
    exec 3>&-
    exec 3<&-
    [[ "$status_line" == *" 101 "* ]]
}

cleanup() {
    if [[ -n "$SERVERS_TMP" ]]; then
        rm -f -- "$SERVERS_TMP"
    fi
    if [[ "$TUNNEL_STARTED" == "1" ]]; then
        if [[ -n "$TUNNEL_CONTROL_PATH" && -S "$TUNNEL_CONTROL_PATH" ]]; then
            ssh -S "$TUNNEL_CONTROL_PATH" -O exit "$TUNNEL_REMOTE" >/dev/null 2>&1 || true
            rm -f -- "$TUNNEL_CONTROL_PATH"
        elif [[ -n "$CONTROL_PATH" && -S "$CONTROL_PATH" && -n "$FORWARD_SPEC" ]]; then
            ssh -S "$CONTROL_PATH" -O cancel -L "$FORWARD_SPEC" "$DISCOVERY_SSH" >/dev/null 2>&1 || true
        fi
    fi
    if [[ "$CLAIM_HELD" == "1" ]]; then
        flock -u 9 >/dev/null 2>&1 || true
        exec 9>&-
        CLAIM_HELD=0
    fi
}

readyz() {
    local line
    if command -v curl >/dev/null 2>&1; then
        curl --noproxy '*' --fail --silent --show-error --max-time 2 \
            "http://127.0.0.1:${LOCAL_PORT}/readyz" >/dev/null 2>&1
        return
    fi
    exec 3<>"/dev/tcp/127.0.0.1/${LOCAL_PORT}" || return 1
    printf 'GET /readyz HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n' >&3
    IFS= read -r line <&3 || line=""
    exec 3>&-
    exec 3<&-
    [[ "$line" == *" 200 "* ]]
}

wait_readyz() {
    local start
    start="$(date +%s)"
    while true; do
        readyz && return
        if (( $(date +%s) - start >= READY_TIMEOUT )); then
            die "SSH tunnel opened, but remote Codex did not answer /readyz"
        fi
        sleep 1
    done
}

usage() {
    cat <<USAGE
usage: bash codex/connect_remote.sh [available|latest|list|list-all|SERVER_ID|NAME|PORT]

Selectors:
  available  newest live server not used by another local connector (default)
  latest     newest live server, even if another local connector uses it
  list       list all server leases
  list-all   include stale and stopped diagnostic leases

Resume:
  CODEX_REMOTE_SESSION_MODE=resume bash codex/connect_remote.sh
  CODEX_REMOTE_SESSION_ID=<thread-id> bash codex/connect_remote.sh

Useful environment:
  CODEX_REMOTE_DISCOVERY_SSH=$DISCOVERY_SSH
  CODEX_REMOTE_ALLOW_REUSE=1
  CODEX_LOCAL_PORT=43121
USAGE
}

case "$SERVER_SELECTOR" in
    help|-h|--help)
        usage
        exit 0
        ;;
esac

case "$SESSION_MODE" in
    new|resume)
        ;;
    *)
        die "CODEX_REMOTE_SESSION_MODE must be 'new' or 'resume'"
        ;;
esac

command -v codex >/dev/null 2>&1 || die "local codex was not found in PATH"
trap cleanup EXIT INT TERM

open_control_master
SERVERS_TMP="$(mktemp)"
read_remote_servers "$SERVERS_TMP"

if [[ "$SERVER_SELECTOR" == "list" || "$SERVER_SELECTOR" == "list-all" ]]; then
    if [[ "$SERVER_SELECTOR" == "list-all" ]]; then
        print_servers 1
    else
        print_servers 0
    fi
    exit 0
fi

select_server
select_local_port "$CODEX_REMOTE_SERVER_PORT"
load_remote_token "$CODEX_REMOTE_SERVER_TOKEN_FILE"
open_tunnel "$CODEX_REMOTE_SERVER_HOST" "$CODEX_REMOTE_DISCOVERY_HOST" "$CODEX_REMOTE_SERVER_BIND_HOST" "$CODEX_REMOTE_SERVER_PORT"
wait_readyz
websocket_auth_is_available || die "remote WebSocket authentication failed"

info "Selected Codex server:"
info "  id:             $CODEX_REMOTE_SERVER_ID"
info "  remote:         $CODEX_REMOTE_SERVER_HOST:$CODEX_REMOTE_SERVER_PORT"
info "  workspace:      $CODEX_REMOTE_SERVER_WORKDIR"
info "  shared history: $CODEX_REMOTE_SERVER_CODEX_HOME"
info "  private SQLite: $CODEX_REMOTE_SERVER_SQLITE_HOME"

if [[ "${CODEX_REMOTE_CONNECT_DRY_RUN:-0}" == "1" ]]; then
    info "Dry run succeeded: ws://127.0.0.1:${LOCAL_PORT}"
    exit 0
fi

common_args=(
    --remote "ws://127.0.0.1:${LOCAL_PORT}"
    -C "$CODEX_REMOTE_SERVER_WORKDIR"
    --sandbox "$SANDBOX"
    --ask-for-approval "$APPROVAL"
)
if [[ -n "$CODEX_REMOTE_APP_TOKEN" ]]; then
    common_args+=(--remote-auth-token-env CODEX_REMOTE_APP_TOKEN)
fi
[[ -n "$MODEL" ]] && common_args+=(-m "$MODEL")

if [[ -n "$SESSION_ID" ]]; then
    cmd=(codex resume "${common_args[@]}")
    [[ "$RESUME_ALL" == "1" ]] && cmd+=(--all)
    cmd+=("$SESSION_ID")
elif [[ "$SESSION_MODE" == "resume" ]]; then
    cmd=(codex resume "${common_args[@]}")
    [[ "$RESUME_ALL" == "1" ]] && cmd+=(--all)
else
    cmd=(codex "${common_args[@]}")
fi

info "Starting local Codex TUI. Disconnecting closes only this tunnel; remote history is already durable."
"${cmd[@]}"
