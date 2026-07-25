#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import remote_history


THREAD_ID = "019f48e4-0006-78f3-b3db-0116c840866a"


def record(value):
    return (json.dumps({"value": value}) + "\n").encode("utf-8")


class RemoteHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.shared = self.root / "shared"
        self.normal = self.root / "normal"
        self.normal.mkdir()
        (self.normal / "auth.json").write_text("{}\n")

    def tearDown(self):
        self.temp.cleanup()

    def rollout(self, home, data):
        path = (
            home
            / "sessions"
            / "2026"
            / "07"
            / "09"
            / ("rollout-test-%s.jsonl" % THREAD_ID)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def migrate(self, roots):
        args = SimpleNamespace(
            shared_home=str(self.shared),
            normal_home=str(self.normal),
            legacy_root=[str(root) for root in roots],
            force=True,
        )
        remote_history.init_history(args)

    def shared_rollout(self):
        paths = list((self.shared / "sessions").rglob("*.jsonl"))
        self.assertEqual(1, len(paths))
        return paths[0]

    def test_imports_and_extends_same_thread(self):
        first = self.root / "first"
        second = self.root / "second"
        self.rollout(first, record(1))
        self.rollout(second, record(1) + record(2))
        self.migrate([first, second])
        self.assertEqual(record(1) + record(2), self.shared_rollout().read_bytes())

    def test_recovers_partial_final_record(self):
        legacy = self.root / "legacy"
        self.rollout(legacy, record(1) + b'{"value":')
        self.migrate([legacy])
        self.assertEqual(record(1), self.shared_rollout().read_bytes())

    def test_preserves_divergent_rollout_as_conflict(self):
        first = self.root / "first"
        second = self.root / "second"
        self.rollout(first, record("a"))
        self.rollout(second, record("b"))
        self.migrate([first, second])
        conflicts = list((self.shared / "recovered_conflicts").rglob("*"))
        self.assertTrue(any(path.is_file() for path in conflicts))


if __name__ == "__main__":
    unittest.main()
