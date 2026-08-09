"""
CleanHouse Utility (scripts/cleanhouse.py)

Pre-flight cleanup utility that checks for stale listener processes on web server ports,
kills orphaned development servers, and verifies environment readiness before start.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import psutil


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP port is currently open and listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _find_pids_via_lsof(port: int) -> list:
    """Fallback listener lookup using lsof (works unprivileged on macOS)."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in out.split() if line.isdigit()]


def find_processes_on_port(port: int) -> list:
    """Find non-system PIDs listening on a given port."""
    pids = []
    try:
        # macOS denies system-wide net_connections() to non-root callers, so this
        # path only succeeds when running elevated; lsof covers the normal case.
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                if conn.pid:
                    pids.append(conn.pid)
    except (psutil.AccessDenied, PermissionError):
        pids = _find_pids_via_lsof(port)
    except Exception:
        pass

    if not pids:
        pids = _find_pids_via_lsof(port)
    return list(set(pids))


def cleanhouse(ports: list[int] = [5000, 5005, 8000, 8080], force: bool = False) -> dict:
    """
    Scan target ports, terminate orphaned python/webserver processes, and return clean ports.

    Args:
        ports: List of target ports to inspect and clean.
        force: If True, force-kill (SIGKILL) unresponsive processes.

    Returns:
        Dictionary detailing closed processes and available clean ports.
    """
    cleaned = []
    skipped_system = []
    available_ports = []

    for port in ports:
        pids = find_processes_on_port(port)
        if not pids:
            # Only advertise the port as clean if nothing is actually listening;
            # an unidentifiable listener still means the port is unusable.
            if not is_port_in_use(port):
                available_ports.append(port)
            continue

        for pid in pids:
            try:
                proc = psutil.Process(pid)
                name = proc.name()

                # Protect macOS system processes like ControlCenter AirPlay Receiver
                if "ControlCenter" in name or "launchd" in name or proc.username() == "root":
                    skipped_system.append({"port": port, "pid": pid, "name": name})
                    continue

                # Terminate stale Python/Flask/Node development servers
                if "python" in name.lower() or "flask" in name.lower() or "node" in name.lower():
                    proc.send_signal(signal.SIGKILL if force else signal.SIGTERM)
                    cleaned.append({"port": port, "pid": pid, "name": name})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # SIGTERM is asynchronous: give the socket a moment to be released.
        for _ in range(20):
            if not is_port_in_use(port):
                break
            time.sleep(0.25)

        if not is_port_in_use(port):
            available_ports.append(port)

    return {
        "success": True,
        "cleaned_processes": cleaned,
        "skipped_system_processes": skipped_system,
        "available_ports": available_ports,
        "recommended_port": available_ports[0] if available_ports else 5005,
    }


if __name__ == "__main__":
    print("🧹 Running CleanHouse Pre-Flight Sweep...")
    report = cleanhouse()
    print(f"Cleaned Processes: {report['cleaned_processes']}")
    print(f"Skipped System Processes: {report['skipped_system_processes']}")
    print(f"Recommended Clean Port: {report['recommended_port']}")
