"""Phase 2 - Hadoop HDFS integration & distributed storage manager.

Wraps the `hdfs dfs` CLI and `hdfs dfsadmin` with graceful dual-mode execution
(actual HDFS when daemons are running, transparent fallback to local staging when offline).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config


def _find_hdfs_binary() -> Optional[str]:
    """Find the hdfs executable in PATH or standard Hadoop install locations."""
    which = shutil.which("hdfs")
    if which:
        return which
    hadoop_home = os.environ.get("HADOOP_HOME", "/opt/homebrew/opt/hadoop/libexec")
    candidate = Path(hadoop_home) / "bin" / "hdfs"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def run_hdfs_cmd(args: List[str], timeout: int = 15) -> subprocess.CompletedProcess:
    """Execute an hdfs CLI command with timeout and return CompletedProcess."""
    hdfs_bin = _find_hdfs_binary()
    if not hdfs_bin:
        raise FileNotFoundError("Hadoop 'hdfs' executable not found in PATH or HADOOP_HOME.")
    cmd = [hdfs_bin, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=(os.name == "nt"),
    )


def check_hdfs_health() -> Dict[str, Any]:
    """Verify live HDFS connectivity, NameNode and DataNode status, and capacity."""
    hdfs_bin = _find_hdfs_binary()
    if not hdfs_bin:
        return {
            "available": False,
            "status": "NOT_INSTALLED",
            "message": "Hadoop 'hdfs' executable not found on system PATH.",
            "namenode_live": False,
            "datanode_live": False,
            "capacity_gb": 0.0,
            "used_mb": 0.0,
            "mode": "Local Fallback",
        }

    # Check JPS for running Java daemons
    jps_bin = shutil.which("jps")
    namenode_up = False
    datanode_up = False
    if jps_bin:
        try:
            res_jps = subprocess.run([jps_bin], capture_output=True, text=True, timeout=5)
            namenode_up = "NameNode" in res_jps.stdout
            datanode_up = "DataNode" in res_jps.stdout
        except Exception:
            pass

    # Check dfsadmin report or dfs ls
    try:
        res = run_hdfs_cmd(["dfsadmin", "-report"], timeout=8)
        if res.returncode == 0:
            output = res.stdout
            capacity_gb = 0.0
            used_mb = 0.0
            for line in output.splitlines():
                if "Configured Capacity:" in line:
                    parts = line.split(":")
                    if len(parts) > 1 and "GB" in parts[1]:
                        try:
                            capacity_gb = float(parts[1].split("(")[1].replace("GB)", "").strip())
                        except Exception:
                            pass
                elif "DFS Used:" in line and "MB" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        try:
                            used_mb = float(parts[1].split("(")[1].replace("MB)", "").strip())
                        except Exception:
                            pass

            return {
                "available": True,
                "status": "HEALTHY",
                "message": f"HDFS cluster live with 1+ DataNodes ({capacity_gb:.1f} GB capacity).",
                "namenode_live": True,
                "datanode_live": True,
                "capacity_gb": capacity_gb,
                "used_mb": used_mb,
                "mode": "HDFS Distributed Storage",
            }
        else:
            # Check if socket connection refused
            msg = res.stderr or res.stdout
            is_offline = "Connection refused" in msg or "ConnectionRefused" in msg or "SocketException" in msg
            return {
                "available": False,
                "status": "DAEMONS_OFFLINE" if is_offline else "DEGRADED",
                "message": "HDFS NameNode is not accepting connections (start via scripts/hadoop_start.sh).",
                "namenode_live": namenode_up,
                "datanode_live": datanode_up,
                "capacity_gb": 0.0,
                "used_mb": 0.0,
                "mode": "Local Fallback",
            }
    except Exception as exc:
        return {
            "available": False,
            "status": "ERROR",
            "message": f"HDFS check encountered error: {str(exc)}",
            "namenode_live": namenode_up,
            "datanode_live": datanode_up,
            "capacity_gb": 0.0,
            "used_mb": 0.0,
            "mode": "Local Fallback",
        }


def upload_to_hdfs(local_path: Optional[Path] = None, hdfs_dir: str = config.HDFS_DIR) -> bool:
    """Upload processed dataset to HDFS with automatic fallback to local staging."""
    src_file = Path(local_path) if local_path else config.PROCESSED_FILE
    if not src_file.exists():
        raise FileNotFoundError(f"Source file for HDFS upload does not exist: {src_file}")

    print(f"[Storage Layer] Ingesting {src_file.name} to storage layer...")
    health = check_hdfs_health()

    if health["available"]:
        print(f"  Live HDFS detected at hdfs://localhost:9000{hdfs_dir}")
        # Create directory
        run_hdfs_cmd(["dfs", "-mkdir", "-p", hdfs_dir])
        target_path = f"{hdfs_dir}/{src_file.name}"
        res = run_hdfs_cmd(["dfs", "-put", "-f", str(src_file), target_path])
        if res.returncode == 0:
            print(f"  Successfully uploaded to HDFS: {target_path}")
            # Verify file exists
            ls_res = run_hdfs_cmd(["dfs", "-ls", hdfs_dir])
            print("  HDFS Directory Listing:\n" + "    " + "\n    ".join(ls_res.stdout.strip().splitlines()))
            return True
        else:
            print(f"  HDFS upload failed ({res.stderr.strip()}), falling back to local storage staging.")

    # Local fallback
    fallback_dir = config.PROC_DIR / "hdfs_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    dest = fallback_dir / src_file.name
    shutil.copy2(src_file, dest)
    print(f"  HDFS daemons offline: Processed data stored in local staging mirror -> {dest}")
    return False


def download_from_hdfs(hdfs_path: str, local_dest: Path) -> bool:
    """Download a file or directory from HDFS to local filesystem."""
    health = check_hdfs_health()
    if not health["available"]:
        fallback_path = config.PROC_DIR / "hdfs_fallback" / Path(hdfs_path).name
        if fallback_path.exists():
            shutil.copy2(fallback_path, local_dest)
            return True
        return False

    res = run_hdfs_cmd(["dfs", "-get", "-f", hdfs_path, str(local_dest)])
    return res.returncode == 0


def list_hdfs_dir(hdfs_dir: str = config.HDFS_DIR) -> List[str]:
    """List files in an HDFS directory."""
    health = check_hdfs_health()
    if not health["available"]:
        fallback_dir = config.PROC_DIR / "hdfs_fallback"
        if fallback_dir.exists():
            return [str(p) for p in fallback_dir.iterdir()]
        return []

    res = run_hdfs_cmd(["dfs", "-ls", hdfs_dir])
    if res.returncode != 0:
        return []
    lines = res.stdout.strip().splitlines()
    files = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 8:
            files.append(parts[-1])
    return files


def main():
    parser = argparse.ArgumentParser(description="Hadoop HDFS Storage Manager")
    parser.add_argument("--status", action="store_true", help="Check HDFS cluster health")
    parser.add_argument("--upload", action="store_true", help="Upload processed bank data to HDFS")
    parser.add_argument("--list", action="store_true", help="List HDFS banking data directory")
    args = parser.parse_args()

    if args.status or (not args.upload and not args.list):
        health = check_hdfs_health()
        print("\n=== HDFS CLUSTER STATUS ===")
        for k, v in health.items():
            print(f"  {k}: {v}")
        print()

    if args.upload:
        upload_to_hdfs()

    if args.list:
        files = list_hdfs_dir()
        print(f"\nFiles in {config.HDFS_DIR}:")
        for f in files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
