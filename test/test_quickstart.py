"""
test_quickstart.py

Location: ./test/test_quickstart.py
Assumes main.py (the FastAPI app) lives in the project root, one level above ./test.

What it does:
1. Starts `ollama run qwen2.5:7b "Hi"` then `uvicorn main:app --reload`
   (run from the root folder, since that's where main.py lives) as a background process.
2. Waits 60 seconds for the server to warm up.
3. Runs `python ./test/test.py`.
4. On exit (normal or Ctrl+C), terminates the server process.
"""

import subprocess
import sys
import time
import os

# Paths
TEST_DIR = os.path.dirname(os.path.abspath(__file__))   # ./test
ROOT_DIR = os.path.dirname(TEST_DIR)                     # project root (where main.py lives)
TEST_SCRIPT = os.path.join(TEST_DIR, "test.py")

# The startup command: ollama warmup, then start uvicorn.
# Using shell=True so command chaining works as in a normal shell.
# NOTE: cmd.exe (Windows default shell) does NOT understand `;` as a separator
# — it treats it as a literal character, which mangles the command line and
# causes errors like "unknown flag: --reload". Use `&` on Windows, `;` on
# POSIX shells (bash/zsh/sh).
CMD_SEPARATOR = "&" if os.name == "nt" else ";"
STARTUP_CMD = f'ollama run qwen2.5:7b "Hi" {CMD_SEPARATOR} uvicorn main:app --reload'

WAIT_SECONDS = 60


def main():
    print(f"[quickstart] Launching server process in {ROOT_DIR!r}:")
    print(f"[quickstart]   {STARTUP_CMD}")

    # Start the long-running process in the background.
    # cwd=ROOT_DIR so `uvicorn main:app` finds main.py correctly.
    server_proc = subprocess.Popen(
        STARTUP_CMD,
        shell=True,
        cwd=ROOT_DIR,
    )

    try:
        print(f"[quickstart] Waiting {WAIT_SECONDS} seconds for the server to be ready...")
        time.sleep(WAIT_SECONDS)

        if server_proc.poll() is not None:
            print(f"[quickstart] WARNING: server process exited early with code {server_proc.returncode}. "
                  f"Continuing anyway, but the test run may fail.")

        print(f"[quickstart] Running tests: python {TEST_SCRIPT}")
        test_result = subprocess.run(
            [sys.executable, TEST_SCRIPT],
            cwd=ROOT_DIR,
        )

        print(f"[quickstart] Tests finished with exit code {test_result.returncode}")
        sys.exit(test_result.returncode)

    except KeyboardInterrupt:
        print("\n[quickstart] Interrupted by user.")

    finally:
        if server_proc.poll() is None:
            print("[quickstart] Shutting down server process...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[quickstart] Server didn't stop in time, killing it.")
                server_proc.kill()


if __name__ == "__main__":
    main()