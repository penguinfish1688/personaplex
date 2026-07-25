#!/usr/bin/env python3
"""Migrate and validate durable Codex rollout history.

Remote app-server processes share one CODEX_HOME so every process sees the
same rollout JSONL files.  SQLite is deliberately *not* merged here: each
server gets its own CODEX_SQLITE_HOME and lets Codex rebuild thread metadata
from the shared rollouts.
"""

from __future__ import print_function

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
SKIP_DIRS = {
    ".git",
    ".cache",
    ".tmp",
    "archived_sessions",
    "codex-node",
    "node",
    "node_modules",
    "runtime",
    "runtimes",
    "servers",
    "sqlite",
    "tmp",
}


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def atomic_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    mkdir(dst.parent)
    tmp = dst.with_name(dst.name + ".tmp.%s" % os.getpid())
    shutil.copy2(str(src), str(tmp))
    os.replace(str(tmp), str(dst))


def atomic_write(path, data):
    path = Path(path)
    mkdir(path.parent)
    tmp = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


class FileLock(object):
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        mkdir(self.path.parent)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def validate_jsonl_bytes(data):
    """Return (valid_bytes, recovered_partial_tail, error)."""
    if not data:
        return None, False, "empty"

    lines = data.splitlines(True)
    valid = []
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped.decode("utf-8"))
        except Exception as exc:
            is_last = index == len(lines) - 1
            has_partial_tail = is_last and not raw_line.endswith((b"\n", b"\r"))
            if has_partial_tail and valid:
                recovered = b"".join(valid)
                if not recovered.endswith(b"\n"):
                    recovered += b"\n"
                return recovered, True, "partial final JSON line: %s" % exc
            return None, False, "invalid JSON line %d: %s" % (index + 1, exc)
        valid.append(raw_line)

    if not valid:
        return None, False, "no JSON records"
    normalized = b"".join(valid)
    if not normalized.endswith(b"\n"):
        normalized += b"\n"
    return normalized, False, None


def thread_id_from_path(path):
    matches = UUID_RE.findall(Path(path).name)
    return matches[-1].lower() if matches else None


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def iter_session_files(root, shared_home):
    root = Path(root).expanduser()
    shared_home = Path(shared_home).resolve()
    if not root.exists():
        return

    if root.name == "sessions":
        session_dirs = [root]
    elif (root / "sessions").is_dir():
        session_dirs = [root / "sessions"]
    else:
        session_dirs = []
        for current, dirs, _files in os.walk(str(root)):
            current_path = Path(current)
            try:
                if current_path.resolve() == shared_home:
                    dirs[:] = []
                    continue
            except OSError:
                pass
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            if current_path.name == "sessions":
                session_dirs.append(current_path)
                dirs[:] = []

    for sessions_dir in session_dirs:
        try:
            if sessions_dir.parent.resolve() == shared_home:
                continue
        except OSError:
            continue
        for path in sessions_dir.rglob("*.jsonl"):
            if path.is_file():
                yield sessions_dir, path


def existing_sessions(shared_home):
    sessions = Path(shared_home) / "sessions"
    by_id = {}
    if not sessions.is_dir():
        return by_id
    for path in sessions.rglob("*.jsonl"):
        thread_id = thread_id_from_path(path)
        if thread_id:
            by_id.setdefault(thread_id, []).append(path)
    return by_id


def default_destination(shared_home, source_sessions, source_path):
    try:
        relative = source_path.relative_to(source_sessions)
        if len(relative.parts) >= 4:
            return Path(shared_home) / "sessions" / relative
    except ValueError:
        pass
    stamp = dt.datetime.fromtimestamp(source_path.stat().st_mtime)
    return (
        Path(shared_home)
        / "sessions"
        / stamp.strftime("%Y")
        / stamp.strftime("%m")
        / stamp.strftime("%d")
        / source_path.name
    )


def conflict_destination(shared_home, thread_id, source_path, data):
    digest = sha256(data)[:12]
    return (
        Path(shared_home)
        / "recovered_conflicts"
        / thread_id
        / (source_path.name + "." + digest)
    )


def copy_or_merge_session(shared_home, source_sessions, source_path, by_id, stats):
    raw = source_path.read_bytes()
    data, recovered, error = validate_jsonl_bytes(raw)
    if data is None:
        stats["invalid"] += 1
        print("warning: skipping invalid rollout %s: %s" % (source_path, error), file=sys.stderr)
        return
    if recovered:
        stats["recovered"] += 1
        print("warning: recovered complete records from %s (%s)" % (source_path, error), file=sys.stderr)

    thread_id = thread_id_from_path(source_path)
    if not thread_id:
        stats["invalid"] += 1
        print("warning: rollout filename has no thread UUID: %s" % source_path, file=sys.stderr)
        return

    destinations = by_id.get(thread_id, [])
    destination = destinations[0] if destinations else default_destination(
        shared_home, source_sessions, source_path
    )

    if destination.exists():
        current_raw = destination.read_bytes()
        current, _recovered, current_error = validate_jsonl_bytes(current_raw)
        if current is None:
            backup = conflict_destination(shared_home, thread_id, destination, current_raw)
            atomic_write(backup, current_raw)
            atomic_write(destination, data)
            stats["replaced_invalid"] += 1
            print(
                "warning: replaced invalid shared rollout %s (%s); backup=%s"
                % (destination, current_error, backup),
                file=sys.stderr,
            )
            return
        if current == data or current.startswith(data):
            stats["unchanged"] += 1
            return
        if data.startswith(current):
            atomic_write(destination, data)
            stats["extended"] += 1
            return

        conflict = conflict_destination(shared_home, thread_id, source_path, data)
        if not conflict.exists():
            atomic_write(conflict, data)
        stats["conflicts"] += 1
        print(
            "warning: divergent rollout for %s preserved at %s" % (thread_id, conflict),
            file=sys.stderr,
        )
        return

    atomic_write(destination, data)
    by_id.setdefault(thread_id, []).append(destination)
    stats["imported"] += 1


def refresh_identity_files(shared_home, normal_home):
    shared_home = Path(shared_home)
    normal_home = Path(normal_home).expanduser()
    auth = normal_home / "auth.json"
    config = normal_home / "config.toml"
    if auth.is_file():
        atomic_copy(auth, shared_home / "auth.json")
        try:
            os.chmod(str(shared_home / "auth.json"), 0o600)
        except OSError:
            pass
    if config.is_file() and not (shared_home / "config.toml").exists():
        atomic_copy(config, shared_home / "config.toml")


def init_history(args):
    shared_home = Path(args.shared_home).expanduser()
    marker = shared_home / ".shared-history-v2.json"
    stats = {
        "conflicts": 0,
        "extended": 0,
        "imported": 0,
        "invalid": 0,
        "recovered": 0,
        "replaced_invalid": 0,
        "unchanged": 0,
    }

    mkdir(shared_home / "sessions")
    with FileLock(shared_home / ".history-migration.lock"):
        refresh_identity_files(shared_home, args.normal_home)
        should_scan = args.force or not marker.exists()
        if should_scan:
            by_id = existing_sessions(shared_home)
            seen_sources = set()
            for root in args.legacy_root:
                for session_dir, source in iter_session_files(root, shared_home) or []:
                    try:
                        source_key = str(source.resolve())
                    except OSError:
                        source_key = str(source)
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    copy_or_merge_session(
                        shared_home, session_dir, source, by_id, stats
                    )

            marker_data = {
                "format": 2,
                "migrated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "legacy_roots": args.legacy_root,
                "stats": stats,
            }
            atomic_write(
                marker,
                (json.dumps(marker_data, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            )

    total = sum(len(paths) for paths in existing_sessions(shared_home).values())
    print(
        "history_ready shared_home=%s sessions=%d imported=%d extended=%d "
        "recovered=%d conflicts=%d invalid=%d"
        % (
            shared_home,
            total,
            stats["imported"],
            stats["extended"],
            stats["recovered"],
            stats["conflicts"],
            stats["invalid"],
        )
    )


def doctor(args):
    shared_home = Path(args.shared_home).expanduser()
    by_id = existing_sessions(shared_home)
    invalid = []
    duplicates = []
    recovered_tail = []
    for thread_id, paths in sorted(by_id.items()):
        if len(paths) > 1:
            duplicates.append((thread_id, paths))
        for path in paths:
            data, recovered, error = validate_jsonl_bytes(path.read_bytes())
            if data is None:
                invalid.append((path, error))
            elif recovered:
                recovered_tail.append((path, error))

    print("shared_home=%s" % shared_home)
    print("threads=%d" % len(by_id))
    print("rollouts=%d" % sum(len(paths) for paths in by_id.values()))
    print("invalid=%d" % len(invalid))
    print("partial_tails=%d" % len(recovered_tail))
    print("duplicate_ids=%d" % len(duplicates))
    for path, error in invalid:
        print("invalid\t%s\t%s" % (path, error))
    for path, error in recovered_tail:
        print("partial_tail\t%s\t%s" % (path, error))
    for thread_id, paths in duplicates:
        print("duplicate\t%s\t%s" % (thread_id, "\t".join(str(path) for path in paths)))
    return 1 if invalid or recovered_tail or duplicates else 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    init_parser = sub.add_parser("init")
    init_parser.add_argument("--shared-home", required=True)
    init_parser.add_argument("--normal-home", required=True)
    init_parser.add_argument("--legacy-root", action="append", default=[])
    init_parser.add_argument("--force", action="store_true")

    doctor_parser = sub.add_parser("doctor")
    doctor_parser.add_argument("--shared-home", required=True)

    args = parser.parse_args(argv)
    if args.command == "init":
        init_history(args)
        return 0
    if args.command == "doctor":
        return doctor(args)
    parser.error("expected init or doctor")


if __name__ == "__main__":
    sys.exit(main())
