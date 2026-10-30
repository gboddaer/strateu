# Run the Python regression tests

These tests need no GPU, model weights or engine build. Use Python 3.11 or newer. The independent portable monitor also has a Python 3.10 CI job.

## Dependencies

Make a separate virtual environment and install `requirements-test.txt`. It includes the pinned runtime packages, Textual/Rich and pytest. Do not use this environment for the **runtime-only** smoke test: that test deliberately rejects environments containing Textual/Rich.

The bundle-import test also needs the real `gguf-py` from the llama.cpp commit pinned in `setup.py`. The PyPI `gguf` package is not a substitute: it lacks the pinned source's `Q2_0` type. This prerequisite is test tooling, not a model download.

From the repository root, on Linux or macOS:

```sh
python3 -m venv .terminal-test-venv
. .terminal-test-venv/bin/activate
python -m pip install -r requirements-test.txt
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git .test-deps/llama.cpp
git -C .test-deps/llama.cpp sparse-checkout set gguf-py
git -C .test-deps/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
export STRATA_GGUF_PY="$PWD/.test-deps/llama.cpp/gguf-py"
```

On Windows, create the environment with `py -3.11 -m venv .terminal-test-venv`, use `.terminal-test-venv\Scripts\python.exe` for the Python commands, and run the same Git commands. Set the path in PowerShell:

```powershell
$env:STRATA_GGUF_PY = "$PWD/.test-deps/llama.cpp/gguf-py"
```

When the installer pin changes, use that new commit here too. The CI regression test checks that the workflow uses the installer pin.

## Suites

```sh
python -m pytest tests/terminal -q --tb=short --junitxml=terminal-results.xml
python -m pytest tools/test_setup_*.py tools/test_strata_mcp.py -q --tb=short --junitxml=setup-results.xml
python -m unittest discover -s serve -p 'test_*.py'
```

The second command uses shell wildcard expansion; on Windows, run it in Git Bash, as CI does. The Terminal suite includes interactive screens driven through Textual's headless test pilot. It takes longer than the setup tests.

The tests use temporary configuration folders. Hardware scenarios should mock the CPU thread count and GPU speeds as well as model names and memory. Colour-specific tests must set or clear `NO_COLOR` themselves. Do not regenerate golden files to match the runner's hardware.

## What CI checks

`broad-tests.yml` runs the full Terminal/Control and setup/MCP suites on Linux and Windows for every PR and every push to `main`. It fetches only the pinned `gguf-py` source, uses read-only repository permissions and keeps JUnit reports for seven days, including failed runs.

The existing minimal-runtime workflow separately checks serving in an environment with **only** `requirements.txt`, then runs the complete server suite. Portable-monitor workflows retain their Python 3.10/3.11 matrix. Hardware inference and native engine builds are separate qualification, not established by these CPU-only tests.
