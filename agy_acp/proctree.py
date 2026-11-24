"""Cross-platform process-tree hygiene for the ``agy acp`` child.

Two rules drive this module:

* a killed parent must never leave an ``agy`` (or a ``cmd.exe`` shim) behind — on Windows a plain
  ``Popen.kill()`` only reaps the direct child, and ``agy.exe`` spawns worker processes;
* nothing here may block a turn for long: every wait is bounded, and the SIGKILL/``taskkill``
  fallback always runs.

POSIX children start in their own session (``start_new_session=True``) so ``killpg`` reaches the
whole tree. Windows children get ``CREATE_NEW_PROCESS_GROUP`` and are killed with
``taskkill /F /T /PID``; when a Job Object can be created (best effort) the child is also assigned
to it, so a crash of this process still reaps the tree.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time

IS_WINDOWS = os.name == "nt"

_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200

_JOB_LOCK = threading.Lock()
_JOB_HANDLES: list[int] = []


def spawn_kwargs() -> dict[str, object]:
    """``subprocess.Popen`` kwargs that make the child a separate, hideable tree root."""
    if IS_WINDOWS:
        flags = _CREATE_NEW_PROCESS_GROUP
        try:  # inside Hermes: same helper the rest of the codebase uses for console hiding
            from hermes_cli._subprocess_compat import windows_hide_flags  # type: ignore

            flags |= int(windows_hide_flags() or 0)
        except Exception:
            flags |= _CREATE_NO_WINDOW
        return {"creationflags": flags}
    return {"start_new_session": True}


def _assign_job_object(proc: subprocess.Popen) -> None:
    """Best-effort Windows Job Object with KILL_ON_JOB_CLOSE (silently skipped elsewhere)."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):  # type: ignore[attr-defined]
            kernel32.CloseHandle(job)
            return
        with _JOB_LOCK:
            _JOB_HANDLES.append(int(job))
    except Exception:
        return


def attach(proc: subprocess.Popen) -> None:
    """Assign a freshly spawned child to its Job Object (Windows only, best effort)."""
    _assign_job_object(proc)


def terminate_tree(proc: subprocess.Popen, *, timeout: float = 3.0) -> None:
    """Terminate ``proc`` and every process it started. Idempotent and never raises."""
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=max(1.0, timeout), check=False,
                creationflags=_CREATE_NO_WINDOW,
            )
        with contextlib.suppress(Exception):
            proc.wait(timeout=timeout)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    if pgid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGTERM)
    else:
        with contextlib.suppress(Exception):
            proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)
    # The direct child dying is not the goal — the whole group has to be empty. Anything still
    # alive after the grace period (a grandchild that ignored SIGTERM) gets SIGKILL.
    deadline = time.monotonic() + timeout
    while pgid is not None and time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.02)
    if pgid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pgid, signal.SIGKILL)
    else:
        with contextlib.suppress(Exception):
            proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)


def is_alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None and proc.stdin is not None and not proc.stdin.closed


def platform_name() -> str:
    return "windows" if IS_WINDOWS else sys.platform
