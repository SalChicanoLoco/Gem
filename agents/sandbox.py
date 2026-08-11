"""
Run a subprocess confined to a single writable directory.

Downloading from the internet is the one thing this app does that is driven by
somebody else's bytes. Even without executing what arrives, a download can write
where it should not: an archive member or a repo filename containing ../ escapes
the destination unless something stops it.

On macOS this uses sandbox-exec, which denies everything by default and then
permits reads, network, and writes only beneath a nominated root. It is the same
Seatbelt mechanism the OS uses for app containers. It is formally deprecated by
Apple but present and working on current systems, so it is treated as a hardening
layer rather than the only defence: callers must still audit paths afterwards
(see web_acquire._audit_containment), because on a platform without sandbox-exec
this degrades to running unconfined and says so.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Seatbelt profile: read anything (the interpreter and its libraries live all
# over the filesystem), reach the network, but write only under write_root plus
# the temp directories Python and requests need to function.
_PROFILE = """(version 1)
(deny default)
(allow process-exec process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow file-read*)
(allow network*)
(allow file-write* (subpath "{write_root}"))
(allow file-write* (subpath "{tmpdir}"))
(allow file-write-data (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))
"""


def available() -> bool:
    """True when subprocess confinement can actually be enforced here."""
    return os.path.exists(SANDBOX_EXEC)


def run_sandboxed(
    argv: List[str],
    write_root: str,
    timeout: int = 900,
    env: Optional[dict] = None,
) -> Tuple[int, str, str, bool]:
    """
    Run argv with writes confined to write_root.

    Returns (returncode, stdout, stderr, confined). `confined` is False when the
    platform offers no sandbox, so a caller can report that honestly instead of
    implying a guarantee it did not get.
    """
    write_root = os.path.abspath(write_root)
    os.makedirs(write_root, exist_ok=True)

    run_env = dict(os.environ)
    if env:
        run_env.update(env)

    if not available():
        logger.warning(
            "sandbox-exec not present; running %s unconfined. Path auditing still applies.",
            argv[0],
        )
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=run_env)
        return proc.returncode, proc.stdout, proc.stderr, False

    tmpdir = tempfile.mkdtemp(prefix="sandbox_tmp_")
    run_env.update({"TMPDIR": tmpdir, "TEMP": tmpdir, "TMP": tmpdir})
    profile_path = os.path.join(tmpdir, "profile.sb")
    try:
        with open(profile_path, "w", encoding="utf-8") as handle:
            handle.write(_PROFILE.format(write_root=write_root, tmpdir=tmpdir))

        proc = subprocess.run(
            [SANDBOX_EXEC, "-f", profile_path] + argv,
            capture_output=True, text=True, timeout=timeout, env=run_env,
        )
        return proc.returncode, proc.stdout, proc.stderr, True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def is_contained(path: str, root: str) -> bool:
    """
    True when path resolves inside root.

    Uses realpath on both sides so a symlink pointing out of the tree is caught
    as well as a literal ../ in a filename.
    """
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    return path_real == root_real or path_real.startswith(root_real + os.sep)
