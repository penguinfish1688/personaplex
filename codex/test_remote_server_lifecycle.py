#!/usr/bin/env python3

import os
import signal
import socket
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = ROOT / "codex" / "start_remote_server.sh"
CONNECTOR_SCRIPT = ROOT / "codex" / "connect_remote.sh"


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def process_exists(pid):
    stat_path = Path("/proc") / str(pid) / "stat"
    if stat_path.exists():
        try:
            if stat_path.read_text().split()[2] == "Z":
                return False
        except (IndexError, OSError):
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class RemoteServerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.remote_root = self.root / "remote"
        self.workspace = self.root / "workspace"
        self.fake_pid = self.root / "fake-server.pid"
        self.fake_child_pid = self.root / "fake-child.pid"
        self.server_process = None
        self.workspace.mkdir()
        self.home.joinpath(".codex").mkdir(parents=True)
        self.home.joinpath(".codex", "auth.json").write_text("{}\n")
        fake_bin = self.home / ".npm-global" / "bin"
        fake_bin.mkdir(parents=True)
        fake_codex = fake_bin / "codex"
        fake_codex.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import http.server
                import os
                import subprocess
                import sys
                from urllib.parse import urlsplit

                listen = sys.argv[sys.argv.index("--listen") + 1]
                parsed = urlsplit(listen)
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(300)"]
                )
                with open(os.environ["FAKE_CODEX_PID_FILE"], "w") as handle:
                    handle.write(str(os.getpid()))
                with open(os.environ["FAKE_CODEX_CHILD_PID_FILE"], "w") as handle:
                    handle.write(str(child.pid))

                class Handler(http.server.BaseHTTPRequestHandler):
                    def do_GET(self):
                        if self.path == "/readyz":
                            self.send_response(200)
                            self.end_headers()
                        else:
                            self.send_response(404)
                            self.end_headers()

                    def log_message(self, _format, *_args):
                        return

                http.server.HTTPServer(
                    (parsed.hostname, parsed.port), Handler
                ).serve_forever()
                """
            )
        )
        fake_codex.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            {
                "HOME": str(self.home),
                "CODEX_REMOTE_WORKDIR": str(self.workspace),
                "CODEX_REMOTE_PROJECT_NAME": "remote_lifecycle_test",
                "CODEX_REMOTE_ROOT": str(self.remote_root),
                "CODEX_REMOTE_SHARED_HOME": str(self.root / "shared-home"),
                "CODEX_REMOTE_HOST": "127.0.0.1",
                "CODEX_REMOTE_BASE_PORT": str(free_port()),
                "CODEX_REMOTE_PORT_SCAN_LIMIT": "1",
                "CODEX_REMOTE_READY_TIMEOUT": "10",
                "CODEX_REMOTE_HEARTBEAT_INTERVAL": "1",
                "CODEX_REMOTE_STALE_SECONDS": "5",
                "CODEX_REMOTE_STOP_WAIT_SECONDS": "10",
                "CODEX_REMOTE_RUNTIME_RETENTION_DAYS": "1",
                "FAKE_CODEX_PID_FILE": str(self.fake_pid),
                "FAKE_CODEX_CHILD_PID_FILE": str(self.fake_child_pid),
                "PYTHONUNBUFFERED": "1",
            }
        )

    def tearDown(self):
        if self.server_process is not None and self.server_process.poll() is None:
            self.server_process.send_signal(signal.SIGTERM)
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait(timeout=5)
        if self.server_process is not None:
            if self.server_process.stdout is not None:
                self.server_process.stdout.close()
            if self.server_process.stderr is not None:
                self.server_process.stderr.close()
        for path in (self.fake_pid, self.fake_child_pid):
            if path.exists():
                pid = int(path.read_text())
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.temp.cleanup()

    def lease_value(self, lease, name):
        command = 'source "$1"; printf "%s" "${!2:-}"'
        return subprocess.check_output(
            ["bash", "-c", command, "lease-value", str(lease), name],
            universal_newlines=True,
        )

    def wait_for_live_server(self):
        self.server_process = subprocess.Popen(
            ["bash", str(SERVER_SCRIPT)],
            cwd=ROOT,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            leases = list(self.remote_root.glob("servers/*.env"))
            if leases:
                lease = leases[0]
                if self.lease_value(lease, "CODEX_REMOTE_SERVER_STATUS") == "live":
                    self.assertTrue(self.fake_pid.exists())
                    self.assertTrue(self.fake_child_pid.exists())
                    return lease
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                self.fail(f"server exited early\nstdout={stdout}\nstderr={stderr}")
            time.sleep(0.1)
        self.fail("server did not become live")

    def assert_server_tree_stopped(self, lease):
        self.server_process.wait(timeout=15)
        self.server_process.stdout.close()
        self.server_process.stderr.close()
        server_pid = int(self.fake_pid.read_text())
        child_pid = int(self.fake_child_pid.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (
            process_exists(server_pid) or process_exists(child_pid)
        ):
            time.sleep(0.1)
        self.assertFalse(process_exists(server_pid))
        self.assertFalse(process_exists(child_pid))
        self.assertEqual(
            "stopped", self.lease_value(lease, "CODEX_REMOTE_SERVER_STATUS")
        )
        heartbeat = Path(
            self.lease_value(lease, "CODEX_REMOTE_SERVER_HEARTBEAT")
        )
        stop_request = Path(
            self.lease_value(lease, "CODEX_REMOTE_SERVER_STOP_REQUEST")
        )
        self.assertFalse(heartbeat.exists())
        self.assertFalse(stop_request.exists())

    def test_stop_request_recycles_server_process_group(self):
        lease = self.wait_for_live_server()
        server_id = self.lease_value(lease, "CODEX_REMOTE_SERVER_ID")
        result = subprocess.run(
            ["bash", str(SERVER_SCRIPT), "stop", server_id, "10"],
            cwd=ROOT,
            env=self.env,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assert_server_tree_stopped(lease)

    def test_hangup_recycles_server_process_group(self):
        lease = self.wait_for_live_server()
        self.server_process.send_signal(signal.SIGHUP)
        self.assert_server_tree_stopped(lease)

    def test_connector_ctrl_c_recycles_selected_server(self):
        self.env["CODEX_REMOTE_HEARTBEAT_INTERVAL"] = "300"
        lease = self.wait_for_live_server()
        server_id = self.lease_value(lease, "CODEX_REMOTE_SERVER_ID")
        time.sleep(0.2)
        lease.write_text(
            "\n".join(
                "CODEX_REMOTE_SERVER_TOKEN_FILE=''"
                if line.startswith("CODEX_REMOTE_SERVER_TOKEN_FILE=")
                else line
                for line in lease.read_text().splitlines()
            )
            + "\n"
        )

        client_bin = self.root / "client-bin"
        client_bin.mkdir()
        marker = self.root / "client.started"
        fake_codex = client_bin / "codex"
        fake_codex.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import os
                import signal
                import sys
                import time

                open(os.environ["FAKE_CLIENT_MARKER"], "w").close()

                def stop(signum, _frame):
                    raise SystemExit(128 + signum)

                signal.signal(signal.SIGINT, stop)
                signal.signal(signal.SIGTERM, stop)
                while True:
                    time.sleep(1)
                """
            )
        )
        fake_codex.chmod(0o755)
        fake_curl = client_bin / "curl"
        fake_curl.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_curl.chmod(0o755)
        fake_ssh = client_bin / "ssh"
        fake_ssh.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail
                mode=""
                while (( $# > 0 )); do
                    case "$1" in
                        -S|-o|-L)
                            shift 2
                            ;;
                        -O)
                            mode="$2"
                            shift 2
                            ;;
                        -*)
                            shift
                            ;;
                        *)
                            shift
                            break
                            ;;
                    esac
                done
                case "$mode" in
                    check|forward|cancel|exit)
                        exit 0
                        ;;
                esac
                if [[ "${1:-}" == "bash -s" ]]; then
                    shift
                    exec bash -s "$@"
                fi
                (( $# > 0 )) || exit 0
                exec "$@"
                """
            )
        )
        fake_ssh.chmod(0o755)

        control_dir = self.root / "ssh-control"
        control_dir.mkdir()
        control_path = control_dir / "fake@login"
        control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control_socket.bind(str(control_path))
        control_socket.close()

        connector_env = self.env.copy()
        connector_env.update(
            {
                "PATH": str(client_bin) + os.pathsep + connector_env["PATH"],
                "CODEX_REMOTE_DISCOVERY_SSH": "fake@login",
                "CODEX_REMOTE_SERVER_ID": server_id,
                "CODEX_REMOTE_SSH_CONTROL_DIR": str(control_dir),
                "CODEX_REMOTE_CLAIM_DIR": str(self.root / "claims"),
                "CODEX_LOCAL_PORT": str(free_port()),
                "CODEX_REMOTE_RECYCLE_TIMEOUT": "10",
                "FAKE_CLIENT_MARKER": str(marker),
            }
        )
        connector = subprocess.Popen(
            ["bash", str(CONNECTOR_SCRIPT)],
            cwd=ROOT,
            env=connector_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not marker.exists():
                if connector.poll() is not None:
                    stdout, stderr = connector.communicate()
                    self.fail(
                        "connector exited early\nstdout={}\nstderr={}".format(
                            stdout, stderr
                        )
                    )
                time.sleep(0.1)
            self.assertTrue(marker.exists(), "local Codex client did not start")
            os.killpg(connector.pid, signal.SIGINT)
            stdout, stderr = connector.communicate(timeout=20)
            self.assertEqual(130, connector.returncode, stderr)
            self.assertIn("Recycling remote Codex server", stderr)
        finally:
            if connector.poll() is None:
                os.killpg(connector.pid, signal.SIGKILL)
                connector.wait(timeout=5)

        self.assert_server_tree_stopped(lease)


if __name__ == "__main__":
    unittest.main()
