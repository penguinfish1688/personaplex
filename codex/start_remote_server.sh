#!/usr/bin/env bash
set -euo pipefail

# Start one Codex app-server for this project.
#
# Every invocation creates a new server process and lease. All servers share
# one CODEX_HOME (durable rollout JSONL, auth, and config), while each process
# gets a private CODEX_SQLITE_HOME. This makes history immediately visible
# across servers without concurrently sharing or merging SQLite databases.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Run this script instead of sourcing it: bash codex/start_remote_server.sh" >&2
    return 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */scripts/remote/codex ]]; then
    DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
else
    DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
fi

PROJECT_ROOT="${CODEX_REMOTE_WORKDIR:-$DEFAULT_PROJECT_ROOT}"
PROJECT_NAME="${CODEX_REMOTE_PROJECT_NAME:-$(basename "$PROJECT_ROOT")}"
REMOTE_ROOT="${CODEX_REMOTE_ROOT:-$HOME/.codex-remote/$PROJECT_NAME}"
SHARED_HOME="${CODEX_REMOTE_SHARED_HOME:-$REMOTE_ROOT/home}"
SERVERS_DIR="$REMOTE_ROOT/servers"
RUNTIMES_DIR="$REMOTE_ROOT/runtimes"
LOCKS_DIR="$REMOTE_ROOT/locks"

HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"
HOSTNAME_SHORT="${HOSTNAME_FQDN%%.*}"

resolve_private_bind_host() {
    local address=""
    if command -v getent >/dev/null 2>&1; then
        address="$(getent ahostsv4 "$HOSTNAME_FQDN" 2>/dev/null | awk '$2 == "STREAM" { print $1; exit }')"
    fi
    if [[ -z "$address" ]]; then
        address="$(hostname -I 2>/dev/null | awk '{ print $1 }')"
    fi
    [[ -n "$address" ]] || return 1
    printf '%s\n' "$address"
}

BIND_HOST="${CODEX_REMOTE_HOST:-$(resolve_private_bind_host)}"
BASE_PORT="${CODEX_REMOTE_BASE_PORT:-43121}"
PORT_SCAN_LIMIT="${CODEX_REMOTE_PORT_SCAN_LIMIT:-100}"
STALE_SECONDS="${CODEX_REMOTE_STALE_SECONDS:-120}"
HEARTBEAT_INTERVAL="${CODEX_REMOTE_HEARTBEAT_INTERVAL:-10}"
READY_TIMEOUT="${CODEX_REMOTE_READY_TIMEOUT:-300}"
RUNTIME_RETENTION_DAYS="${CODEX_REMOTE_RUNTIME_RETENTION_DAYS:-7}"
SERVER_NAME="${CODEX_REMOTE_NAME:-default}"
SERVER_ID="${CODEX_REMOTE_SERVER_ID:-$(date -u +%Y%m%dT%H%M%SZ).${HOSTNAME_SHORT}.$$}"
SERVER_RUNTIME_DIR="$RUNTIMES_DIR/$SERVER_ID"
SERVER_SQLITE_HOME="${CODEX_REMOTE_SQLITE_HOME:-$SERVER_RUNTIME_DIR/sqlite}"
SERVER_TOKEN_FILE="${CODEX_REMOTE_TOKEN_FILE:-$SERVER_RUNTIME_DIR/app-server.token}"
LEASE_FILE="$SERVERS_DIR/$SERVER_ID.env"
HEARTBEAT_FILE="$SERVERS_DIR/$SERVER_ID.heartbeat"
START_LOCK="$LOCKS_DIR/start.${HOSTNAME_SHORT}.lockdir"

PORT=""
APP_SERVER_PID=""
HEARTBEAT_PID=""
SERVER_STATUS="starting"
START_LOCK_HELD=0
LEASE_WRITTEN=0
FINALIZED=0
STARTED_AT=""

info() {
    printf '%s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

quote_env() {
    printf '%q' "$1"
}

valid_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

validate_positive_int() {
    local value="$1"
    local name="$2"
    valid_int "$value" || die "$name must be an integer, got: $value"
    (( value > 0 )) || die "$name must be positive, got: $value"
}

now_epoch() {
    date +%s
}

age_seconds() {
    local path="$1"
    local mtime
    [[ -e "$path" ]] || {
        printf '999999999\n'
        return
    }
    mtime="$(stat -c '%Y' "$path" 2>/dev/null || printf '0')"
    printf '%s\n' "$(( $(now_epoch) - mtime ))"
}

prepare_path() {
    local node_path
    local codex_path
    local dir

    export PATH="$HOME/.npm-global/bin:$HOME/.conda/envs/codex-node/bin:$SCRIPT_DIR/codex-node/bin:$SCRIPT_DIR/node/bin:$PATH"

    for node_path in \
        "$HOME/.conda/envs/codex-node/bin/node" \
        "$HOME/.npm-global/bin/node" \
        "$SCRIPT_DIR/codex-node/bin/node" \
        "$SCRIPT_DIR/node/bin/node" \
        "$HOME"/.nvm/versions/node/*/bin/node; do
        if [[ -x "$node_path" ]]; then
            export PATH="$(dirname "$node_path"):$PATH"
            break
        fi
    done

    for codex_path in \
        "$HOME/.npm-global/bin/codex" \
        "$SCRIPT_DIR/codex-node/bin/codex" \
        "$SCRIPT_DIR/node/bin/codex"; do
        if [[ -x "$codex_path" ]]; then
            export PATH="$(dirname "$codex_path"):$PATH"
            break
        fi
    done

    for dir in \
        "$SCRIPT_DIR"/codex-node/lib/node_modules/@openai/codex/node_modules/@openai/codex-*/vendor/*/codex-resources \
        "$HOME"/.npm-global/lib/node_modules/@openai/codex/node_modules/@openai/codex-*/vendor/*/codex-resources; do
        if [[ -x "$dir/bwrap" ]]; then
            export PATH="$dir:$PATH"
            break
        fi
    done

    command -v python3 >/dev/null 2>&1 || die "python3 was not found"
    command -v node >/dev/null 2>&1 || die "node was not found; expected ~/.conda/envs/codex-node/bin/node"
    command -v codex >/dev/null 2>&1 || die "codex was not found; expected ~/.npm-global/bin/codex"
}

generate_server_token() {
    local token_tmp
    mkdir -p "$(dirname "$SERVER_TOKEN_FILE")"
    token_tmp="$(mktemp "${SERVER_TOKEN_FILE}.tmp.XXXXXX")"
    python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$token_tmp"
    chmod 600 "$token_tmp"
    mv "$token_tmp" "$SERVER_TOKEN_FILE"
}

history_init() {
    local args
    [[ -f "$SCRIPT_DIR/remote_history.py" ]] || die "missing $SCRIPT_DIR/remote_history.py"

    args=(
        init
        --shared-home "$SHARED_HOME"
        --normal-home "${CODEX_REMOTE_AUTH_HOME:-$HOME/.codex}"
        --legacy-root "$REMOTE_ROOT/homes"
        --legacy-root "$PROJECT_ROOT/codex"
        --legacy-root "$PROJECT_ROOT/.codex-remote"
    )
    if [[ "${CODEX_REMOTE_MIGRATE_HISTORY:-auto}" == "force" ]]; then
        args+=(--force)
    fi
    python3 "$SCRIPT_DIR/remote_history.py" "${args[@]}"
}

history_doctor() {
    python3 "$SCRIPT_DIR/remote_history.py" doctor --shared-home "$SHARED_HOME"
}

port_is_listening() {
    local port="$1"
    (exec 3<>"/dev/tcp/${BIND_HOST}/${port}") >/dev/null 2>&1
}

readyz() {
    local port="$1"
    local line
    if command -v curl >/dev/null 2>&1; then
        curl --noproxy '*' --fail --silent --show-error --max-time 2 \
            "http://${BIND_HOST}:${port}/readyz" >/dev/null 2>&1
        return
    fi
    exec 3<>"/dev/tcp/${BIND_HOST}/${port}" || return 1
    printf 'GET /readyz HTTP/1.0\r\nHost: %s\r\n\r\n' "$BIND_HOST" >&3
    IFS= read -r line <&3 || line=""
    exec 3>&-
    exec 3<&-
    [[ "$line" == *" 200 "* ]]
}

acquire_start_lock() {
    local age
    local owner_pid=""
    mkdir -p "$LOCKS_DIR"
    while ! mkdir "$START_LOCK" 2>/dev/null; do
        if [[ -r "$START_LOCK/pid" ]]; then
            owner_pid="$(<"$START_LOCK/pid")"
        fi
        age="$(age_seconds "$START_LOCK")"
        if (( age > STALE_SECONDS )) &&
            { [[ ! "$owner_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$owner_pid" >/dev/null 2>&1; }; then
            rm -rf -- "$START_LOCK"
            continue
        fi
        sleep 1
    done
    START_LOCK_HELD=1
    printf '%s\n' "$$" > "$START_LOCK/pid"
}

release_start_lock() {
    if [[ "$START_LOCK_HELD" == "1" ]]; then
        rm -rf -- "$START_LOCK"
        START_LOCK_HELD=0
    fi
}

select_port() {
    local requested="${CODEX_REMOTE_PORT:-}"
    local candidate
    local last_port

    validate_positive_int "$BASE_PORT" CODEX_REMOTE_BASE_PORT
    validate_positive_int "$PORT_SCAN_LIMIT" CODEX_REMOTE_PORT_SCAN_LIMIT

    if [[ -n "$requested" ]]; then
        validate_positive_int "$requested" CODEX_REMOTE_PORT
        (( requested <= 65535 )) || die "CODEX_REMOTE_PORT must be <= 65535"
        port_is_listening "$requested" && die "requested port $requested is already in use on $HOSTNAME_FQDN"
        PORT="$requested"
        return
    fi

    last_port=$(( BASE_PORT + PORT_SCAN_LIMIT - 1 ))
    (( last_port > 65535 )) && last_port=65535
    for ((candidate = BASE_PORT; candidate <= last_port; candidate++)); do
        if ! port_is_listening "$candidate"; then
            PORT="$candidate"
            return
        fi
    done
    die "no free port found in ${BASE_PORT}-${last_port} on $HOSTNAME_FQDN"
}

write_lease() {
    local tmp
    mkdir -p "$SERVERS_DIR"
    tmp="$(mktemp "$LEASE_FILE.tmp.XXXXXX")"
    {
        printf 'CODEX_REMOTE_SERVER_ID=%s\n' "$(quote_env "$SERVER_ID")"
        printf 'CODEX_REMOTE_SERVER_NAME=%s\n' "$(quote_env "$SERVER_NAME")"
        printf 'CODEX_REMOTE_SERVER_STATUS=%s\n' "$(quote_env "$SERVER_STATUS")"
        printf 'CODEX_REMOTE_SERVER_HOST=%s\n' "$(quote_env "$HOSTNAME_FQDN")"
        printf 'CODEX_REMOTE_SERVER_BIND_HOST=%s\n' "$(quote_env "$BIND_HOST")"
        printf 'CODEX_REMOTE_SERVER_PORT=%s\n' "$(quote_env "$PORT")"
        printf 'CODEX_REMOTE_SERVER_WORKDIR=%s\n' "$(quote_env "$PROJECT_ROOT")"
        printf 'CODEX_REMOTE_SERVER_CODEX_HOME=%s\n' "$(quote_env "$SHARED_HOME")"
        printf 'CODEX_REMOTE_SERVER_SQLITE_HOME=%s\n' "$(quote_env "$SERVER_SQLITE_HOME")"
        printf 'CODEX_REMOTE_SERVER_TOKEN_FILE=%s\n' "$(quote_env "$SERVER_TOKEN_FILE")"
        printf 'CODEX_REMOTE_SERVER_RUNTIME_DIR=%s\n' "$(quote_env "$SERVER_RUNTIME_DIR")"
        printf 'CODEX_REMOTE_SERVER_PID=%s\n' "$(quote_env "$$")"
        printf 'CODEX_REMOTE_SERVER_APP_PID=%s\n' "$(quote_env "${APP_SERVER_PID:-}")"
        printf 'CODEX_REMOTE_SERVER_HEARTBEAT=%s\n' "$(quote_env "$HEARTBEAT_FILE")"
        printf 'CODEX_REMOTE_SERVER_ROOT=%s\n' "$(quote_env "$REMOTE_ROOT")"
        printf 'CODEX_REMOTE_SERVER_STARTED_AT=%s\n' "$(quote_env "${STARTED_AT:-$(date -Is)}")"
        printf 'CODEX_REMOTE_SERVER_UPDATED_AT=%s\n' "$(quote_env "$(date -Is)")"
    } > "$tmp"
    mv "$tmp" "$LEASE_FILE"
    LEASE_WRITTEN=1
}

heartbeat_loop() {
    while true; do
        date +%s > "$HEARTBEAT_FILE"
        write_lease
        sleep "$HEARTBEAT_INTERVAL"
    done
}

lease_is_fresh() {
    local heartbeat="$1"
    [[ -n "$heartbeat" && -e "$heartbeat" ]] || return 1
    (( $(age_seconds "$heartbeat") <= STALE_SECONDS ))
}

prune_stale_runtimes() {
    local archive_dir="$REMOTE_ROOT/archive/leases"
    local lease
    local retention_seconds
    local runtime
    local status
    local heartbeat
    local prune_now="${CODEX_REMOTE_PRUNE_NOW:-0}"

    validate_positive_int "$RUNTIME_RETENTION_DAYS" CODEX_REMOTE_RUNTIME_RETENTION_DAYS
    retention_seconds=$(( RUNTIME_RETENTION_DAYS * 86400 ))
    shopt -s nullglob
    for lease in "$SERVERS_DIR"/*.env; do
        unset CODEX_REMOTE_SERVER_STATUS CODEX_REMOTE_SERVER_HEARTBEAT CODEX_REMOTE_SERVER_RUNTIME_DIR
        # shellcheck disable=SC1090
        source "$lease" || continue
        status="${CODEX_REMOTE_SERVER_STATUS:-unknown}"
        heartbeat="${CODEX_REMOTE_SERVER_HEARTBEAT:-}"
        runtime="${CODEX_REMOTE_SERVER_RUNTIME_DIR:-}"
        if [[ "$status" == "live" || "$status" == "starting" ]] && lease_is_fresh "$heartbeat"; then
            continue
        fi
        if [[ "$prune_now" != "1" ]]; then
            (( $(age_seconds "$lease") > retention_seconds )) || continue
        fi
        if [[ -n "$runtime" && -f "$runtime/.codex-remote-runtime" ]]; then
            rm -rf -- "$runtime"
        fi
        mkdir -p "$archive_dir"
        if [[ -n "$heartbeat" && -e "$heartbeat" ]]; then
            mv "$heartbeat" "$archive_dir/$(basename "$heartbeat").$(date +%s)" 2>/dev/null || true
        fi
        mv "$lease" "$archive_dir/$(basename "$lease")" 2>/dev/null || true
    done
    shopt -u nullglob
}

seed_sqlite_from_stopped_runtime() {
    local lease
    local newest_lease=""
    local newest_mtime=0
    local lease_mtime
    local runtime
    local sqlite_home
    local status

    [[ "${CODEX_REMOTE_REUSE_SQLITE:-1}" == "1" ]] || return 0
    [[ -z "$(find "$SERVER_SQLITE_HOME" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || return 0

    shopt -s nullglob
    for lease in "$SERVERS_DIR"/*.env; do
        unset CODEX_REMOTE_SERVER_STATUS CODEX_REMOTE_SERVER_RUNTIME_DIR CODEX_REMOTE_SERVER_SQLITE_HOME
        # shellcheck disable=SC1090
        source "$lease" || continue
        status="${CODEX_REMOTE_SERVER_STATUS:-unknown}"
        runtime="${CODEX_REMOTE_SERVER_RUNTIME_DIR:-}"
        sqlite_home="${CODEX_REMOTE_SERVER_SQLITE_HOME:-}"
        [[ "$status" == "stopped" ]] || continue
        [[ -n "$runtime" && -f "$runtime/.codex-remote-runtime" ]] || continue
        [[ -n "$sqlite_home" && -d "$sqlite_home" ]] || continue
        [[ -n "$(find "$sqlite_home" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || continue
        lease_mtime="$(stat -c '%Y' "$lease" 2>/dev/null || printf '0')"
        if (( lease_mtime > newest_mtime )); then
            newest_mtime="$lease_mtime"
            newest_lease="$lease"
        fi
    done
    shopt -u nullglob

    [[ -n "$newest_lease" ]] || return 0
    unset CODEX_REMOTE_SERVER_SQLITE_HOME
    # shellcheck disable=SC1090
    source "$newest_lease"
    sqlite_home="$CODEX_REMOTE_SERVER_SQLITE_HOME"
    info "Reusing SQLite index from stopped server $(basename "$newest_lease" .env)."
    cp -a -- "$sqlite_home/." "$SERVER_SQLITE_HOME/"
}

wait_until_ready() {
    local elapsed
    local next_notice=15
    local start
    start="$(now_epoch)"
    while true; do
        readyz "$PORT" && return 0
        if [[ -n "$APP_SERVER_PID" ]] && ! kill -0 "$APP_SERVER_PID" >/dev/null 2>&1; then
            wait "$APP_SERVER_PID" || true
            die "codex app-server exited before /readyz"
        fi
        elapsed=$(( $(now_epoch) - start ))
        if (( elapsed >= READY_TIMEOUT )); then
            info "Codex app-server process at readiness timeout:"
            ps -o pid=,ppid=,stat=,etime=,args= -p "$APP_SERVER_PID" >&2 || true
            die "codex app-server did not answer /readyz within ${READY_TIMEOUT}s"
        fi
        if (( elapsed >= next_notice )); then
            info "Still waiting for Codex /readyz (${elapsed}s). Building a fresh SQLite index for a large shared history can take several minutes."
            next_notice=$(( next_notice + 30 ))
        fi
        sleep 1
    done
}

finalize() {
    [[ "$FINALIZED" == "0" ]] || return 0
    FINALIZED=1

    if [[ -n "$HEARTBEAT_PID" ]] && kill -0 "$HEARTBEAT_PID" >/dev/null 2>&1; then
        kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true
        wait "$HEARTBEAT_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$APP_SERVER_PID" ]] && kill -0 "$APP_SERVER_PID" >/dev/null 2>&1; then
        kill "$APP_SERVER_PID" >/dev/null 2>&1 || true
        for _ in {1..50}; do
            kill -0 "$APP_SERVER_PID" >/dev/null 2>&1 || break
            sleep 0.1
        done
        if kill -0 "$APP_SERVER_PID" >/dev/null 2>&1; then
            kill -KILL "$APP_SERVER_PID" >/dev/null 2>&1 || true
        fi
        wait "$APP_SERVER_PID" >/dev/null 2>&1 || true
    fi
    APP_SERVER_PID=""
    release_start_lock
    if [[ "$LEASE_WRITTEN" == "1" ]]; then
        SERVER_STATUS="stopped"
        write_lease || true
    fi
    rm -f -- "$HEARTBEAT_FILE"
}

stop_server() {
    trap - INT TERM
    if [[ -n "$APP_SERVER_PID" ]] && kill -0 "$APP_SERVER_PID" >/dev/null 2>&1; then
        kill "$APP_SERVER_PID" >/dev/null 2>&1 || true
        wait "$APP_SERVER_PID" >/dev/null 2>&1 || true
    fi
    finalize
    info
    info "Codex app server stopped. Shared history remains at $SHARED_HOME."
    exit 0
}

status() {
    local show_all="${1:-0}"
    local lease
    local count=0
    local freshness
    shopt -s nullglob
    for lease in "$SERVERS_DIR"/*.env; do
        unset CODEX_REMOTE_SERVER_ID CODEX_REMOTE_SERVER_NAME CODEX_REMOTE_SERVER_STATUS
        unset CODEX_REMOTE_SERVER_HOST CODEX_REMOTE_SERVER_PORT CODEX_REMOTE_SERVER_HEARTBEAT
        unset CODEX_REMOTE_SERVER_CODEX_HOME CODEX_REMOTE_SERVER_SQLITE_HOME
        # shellcheck disable=SC1090
        source "$lease" || continue
        freshness="stale"
        lease_is_fresh "${CODEX_REMOTE_SERVER_HEARTBEAT:-}" && freshness="fresh"
        if [[ "$show_all" != "1" ]] &&
            { [[ "${CODEX_REMOTE_SERVER_STATUS:-}" != "live" ]] || [[ "$freshness" != "fresh" ]]; }; then
            continue
        fi
        printf '%s\t%s\t%s\t%s\t%s:%s\tshared=%s\tsqlite=%s\n' \
            "${CODEX_REMOTE_SERVER_ID:-unknown}" \
            "${CODEX_REMOTE_SERVER_NAME:-default}" \
            "${CODEX_REMOTE_SERVER_STATUS:-unknown}" \
            "$freshness" \
            "${CODEX_REMOTE_SERVER_HOST:-unknown}" \
            "${CODEX_REMOTE_SERVER_PORT:-unknown}" \
            "${CODEX_REMOTE_SERVER_CODEX_HOME:-unknown}" \
            "${CODEX_REMOTE_SERVER_SQLITE_HOME:-unknown}"
        count=$((count + 1))
    done
    shopt -u nullglob
    (( count > 0 )) || info "No Codex server leases for '$PROJECT_NAME'."
}

case "${1:-start}" in
    start)
        ;;
    status|list)
        status 0
        exit 0
        ;;
    status-all|list-all)
        status 1
        exit 0
        ;;
    doctor)
        prepare_path
        history_doctor
        exit $?
        ;;
    migrate-history)
        prepare_path
        CODEX_REMOTE_MIGRATE_HISTORY=force history_init
        history_doctor
        exit $?
        ;;
    prune)
        CODEX_REMOTE_PRUNE_NOW=1 prune_stale_runtimes
        exit 0
        ;;
    *)
        die "usage: $0 [start|status|status-all|doctor|migrate-history|prune]"
        ;;
esac

cd "$PROJECT_ROOT"
prepare_path
validate_positive_int "$STALE_SECONDS" CODEX_REMOTE_STALE_SECONDS
validate_positive_int "$HEARTBEAT_INTERVAL" CODEX_REMOTE_HEARTBEAT_INTERVAL
validate_positive_int "$READY_TIMEOUT" CODEX_REMOTE_READY_TIMEOUT

mkdir -p "$REMOTE_ROOT" "$SHARED_HOME" "$SERVERS_DIR" "$RUNTIMES_DIR" "$LOCKS_DIR"
history_init
prune_stale_runtimes

mkdir -p "$SERVER_SQLITE_HOME"
printf '%s\n' "$SERVER_ID" > "$SERVER_RUNTIME_DIR/.codex-remote-runtime"
trap finalize EXIT
trap stop_server INT TERM
acquire_start_lock
seed_sqlite_from_stopped_runtime
generate_server_token
export CODEX_HOME="$SHARED_HOME"
export CODEX_SQLITE_HOME="$SERVER_SQLITE_HOME"

if [[ ! -s "$CODEX_HOME/auth.json" && -z "${OPENAI_API_KEY:-}" && -z "${CODEX_ACCESS_TOKEN:-}" ]]; then
    die "no Codex auth found at $CODEX_HOME/auth.json; run codex login on the remote host"
fi

select_port
STARTED_AT="$(date -Is)"
SERVER_STATUS="starting"
write_lease

info "Starting Codex app server."
info "  project:        $PROJECT_NAME"
info "  workspace:      $PROJECT_ROOT"
info "  remote:         ws://$BIND_HOST:$PORT"
info "  server id:      $SERVER_ID"
info "  shared history: $SHARED_HOME"
info "  private SQLite: $SERVER_SQLITE_HOME"
info "  WebSocket auth: capability token"
info "  lease:          $LEASE_FILE"
info
info "Connect locally with: bash $SCRIPT_DIR/connect_remote.sh"
info "List servers with:    bash $SCRIPT_DIR/start_remote_server.sh status"
info "Keep this terminal open while the server should remain available."

codex app-server \
    -c "sqlite_home=\"$SERVER_SQLITE_HOME\"" \
    --listen "ws://${BIND_HOST}:${PORT}" \
    --ws-auth capability-token \
    --ws-token-file "$SERVER_TOKEN_FILE" &
APP_SERVER_PID=$!
wait_until_ready
SERVER_STATUS="live"
write_lease
date +%s > "$HEARTBEAT_FILE"
heartbeat_loop &
HEARTBEAT_PID=$!
release_start_lock

set +e
wait "$APP_SERVER_PID"
STATUS=$?
set -e
APP_SERVER_PID=""
finalize

if [[ "$STATUS" -eq 130 || "$STATUS" -eq 143 ]]; then
    exit 0
fi
exit "$STATUS"
