"""The engine's `RESUME` line, end to end through StrataEngine (no GPU, no model pack).

Why this file exists: `serve/server.py` parses `RESUME <n>` (:1934) to learn how much of a prompt the engine did
not read, and `request_timings` turns that into `cache_n`.  Every other test set `reused` on a fake engine object,
so the LINE itself had no non-zero coverage - the engine could stop printing it, or print it in the wrong place,
and the suite stayed green.  This drives a stub engine that speaks the real protocol.

    python -m unittest serve.test_resume_protocol -v
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.server import StrataEngine, request_timings

# A stub engine: READY, then for each GEN a RESUME line, one PP line, one token, and a DONE carrying the reuse in
# f[8] and the tokens actually read in f[14] - the two fields serve/server.py reads.
FAKE = r'''
import sys
print("READY 8192 stop", flush=True)
for line in sys.stdin:
    if not line.startswith("GEN"):
        continue
    print("RESUME 516", flush=True)
    print("PP 517 580 120.0 500.0", flush=True)
    print("T 42", flush=True)
    print("DONE 1 580 120.0 90.0 length 0 0 516 0 0 0 0 0.0 64 0", flush=True)
'''


class ResumeProtocol(unittest.TestCase):
    def test_resume_line_carries_the_reused_count_into_the_timings(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "fake_engine.py"
        script.write_text(FAKE, encoding="utf-8")
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            engine = StrataEngine("strata", [])
        self.addCleanup(engine.unload)

        # a None heartbeat precedes the tokens (the server uses it for its 10 s keep-alive)
        tokens = [t for t in engine.generate([1, 2, 3], 1, {}, threading.Event()) if t is not None]

        self.assertEqual(tokens, [42])
        self.assertEqual(engine.last["reused"], 516, "the RESUME line is what sets this")
        self.assertEqual(engine.last["prompt_read"], 64, "the DONE's f[14]")
        timings = request_timings(580, 1, engine.last)
        self.assertEqual((timings["cache_n"], timings["prompt_n"]), (516, 64))
        self.assertEqual(timings["cache_n"] + timings["prompt_n"], 580)


if __name__ == "__main__":
    unittest.main()
