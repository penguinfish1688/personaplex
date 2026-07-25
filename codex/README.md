# Remote Codex on ORCD

These helpers run multiple independent Codex app servers for the same ORCD
checkout and connect local Codex TUIs through SSH tunnels.

## Start servers on ORCD

Each invocation starts one new server and stays in the foreground:

```bash
bash ~/personaplex/codex/start_remote_server.sh
```

The private Git checkout stores these scripts at
`~/personaplex/personaplex/codex`. Expose them at the aggregate workspace root
once so the launcher uses `~/personaplex` as its default workdir:

```bash
ln -s personaplex/codex ~/personaplex/codex
```

Run the command in another ORCD terminal for every additional server/agent you
want. The script selects a free port and prints its unique server id.

List live server leases (use `status-all` for stale diagnostics):

```bash
bash ~/personaplex/codex/start_remote_server.sh status
bash ~/personaplex/codex/start_remote_server.sh status-all
```

## Connect from penguinfish

The default selector chooses the newest live server not already used by another
local connector:

```bash
cd ~/personaplex
bash codex/connect_remote.sh
```

Run that local command again to connect another TUI to another available remote
server. You can also list or select a server explicitly:

```bash
bash codex/connect_remote.sh list
bash codex/connect_remote.sh list-all
bash codex/connect_remote.sh SERVER_ID
bash codex/connect_remote.sh PORT
bash codex/connect_remote.sh NAME
```

`latest` intentionally allows reuse of the newest server. The equivalent
environment override is `CODEX_REMOTE_ALLOW_REUSE=1`.

```bash
bash codex/connect_remote.sh latest
```

## Resume history

Inside the connected TUI, `/resume` sees project history written by every
server. You can also open the resume picker immediately or target a thread id:

```bash
CODEX_REMOTE_SESSION_MODE=resume bash codex/connect_remote.sh
CODEX_REMOTE_SESSION_ID=THREAD_ID bash codex/connect_remote.sh
```

Disconnecting the local TUI closes only its SSH tunnel. Transcript JSONL is
already stored on ORCD and does not need a shutdown-time copy or merge.

## State layout

All servers use:

```text
~/.codex-remote/personaplex/home/
```

as their shared `CODEX_HOME`. This contains durable sessions, auth, and config.
Each live server uses a separate SQLite directory:

```text
~/.codex-remote/personaplex/runtimes/SERVER_ID/sqlite/
```

This split is intentional: shared rollout files make history immediately
visible, while private SQLite databases avoid cross-process database merging and
locking problems. A new server copies the newest cleanly stopped server's
private database and Codex updates it incrementally. It only builds an index
from scratch when no safely reusable database exists.

Legacy writer homes are imported once into the shared session tree. They are
left in place as backups; no old database is merged into a live database.

Validate or rerun history migration on ORCD:

```bash
bash ~/personaplex/codex/start_remote_server.sh doctor
bash ~/personaplex/codex/start_remote_server.sh migrate-history
```

Stopped per-server SQLite runtimes are retained for seven days by default and
then pruned on a later server start. To prune eligible runtimes immediately:

```bash
bash ~/personaplex/codex/start_remote_server.sh prune
```

## Useful overrides

```bash
CODEX_REMOTE_NAME=agent1 bash codex/start_remote_server.sh
CODEX_REMOTE_PORT=43130 bash codex/start_remote_server.sh
CODEX_REMOTE_READY_TIMEOUT=600 bash codex/start_remote_server.sh
CODEX_REMOTE_RUNTIME_RETENTION_DAYS=14 bash codex/start_remote_server.sh
CODEX_REMOTE_REUSE_SQLITE=0 bash codex/start_remote_server.sh
CODEX_LOCAL_PORT=44130 bash codex/connect_remote.sh agent1
CODEX_REMOTE_RESET_SSH=1 bash codex/connect_remote.sh
```

A server normally seeds its private SQLite directory from the newest cleanly
stopped server, avoiding a full history rebuild on restart. If no clean database
is available, the initial index can take several minutes on ORCD shared storage.
Set `CODEX_REMOTE_REUSE_SQLITE=0` to force a fresh index, or override
`CODEX_REMOTE_READY_TIMEOUT` when a fresh index needs longer.

`WARNING: failed to clean up stale arg0 temp dirs: Directory not empty` is a
best-effort Codex helper cleanup warning and does not stop app-server startup.

Each server discovers and records its current login node and private ORCD IP at
startup. The connector also discovers whichever login node the public ORCD SSH
connection currently reaches, then uses that one control connection to forward
to the selected server's recorded private IP. No `loginNNN` hostname is
hard-coded and no second SSH login to `loginNNN.inband` is needed.

The first connector invocation creates the control channel with a brief ORCD
login shell after the password and Duo prompt. It automatically exits that
shell and keeps the authenticated connection alive with `ControlPersist`. If an
old socket exists but cannot open commands, the connector discards it and
performs a fresh public ORCD login automatically.

Every WebSocket listener requires a unique capability token. The connector
reads that token through the existing SSH control connection and supplies it to
the local Codex TUI. See the official Codex app-server documentation:
<https://developers.openai.com/codex/app-server>.
