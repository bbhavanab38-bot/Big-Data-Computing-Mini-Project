"""Tests for HDFS distributed storage manager and health diagnostics."""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.hdfs_utils import check_hdfs_health, download_from_hdfs, list_hdfs_dir, upload_to_hdfs


def test_check_hdfs_health_structure():
    """Ensure check_hdfs_health returns all required status dictionary keys."""
    health = check_hdfs_health()
    assert isinstance(health, dict)
    assert "available" in health
    assert "status" in health
    assert "message" in health
    assert "namenode_live" in health
    assert "datanode_live" in health
    assert "capacity_gb" in health
    assert "used_mb" in health
    assert "mode" in health
    assert isinstance(health["available"], bool)


def test_check_hdfs_health_when_not_installed():
    """When hdfs CLI is missing, health check returns NOT_INSTALLED and Local Fallback."""
    with patch("src.hdfs_utils._find_hdfs_binary", return_value=None):
        health = check_hdfs_health()
        assert health["available"] is False
        assert health["status"] == "NOT_INSTALLED"
        assert health["mode"] == "Local Fallback"


def test_check_hdfs_health_when_connection_refused():
    """When daemons are offline, health check returns DAEMONS_OFFLINE."""
    mock_res = MagicMock()
    mock_res.returncode = 1
    mock_res.stderr = "Call to localhost:9000 failed on connection exception: Connection refused"
    mock_res.stdout = ""

    with patch("src.hdfs_utils._find_hdfs_binary", return_value="/opt/homebrew/bin/hdfs"):
        with patch("src.hdfs_utils.run_hdfs_cmd", return_value=mock_res):
            health = check_hdfs_health()
            assert health["available"] is False
            assert health["status"] == "DAEMONS_OFFLINE"
            assert "not accepting connections" in health["message"]


def test_upload_to_hdfs_fallback_when_offline(tmp_path):
    """When HDFS is offline, upload_to_hdfs cleanly copies file to local fallback staging."""
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("CustomerId,Exited\n1,0\n")

    with patch("src.hdfs_utils.check_hdfs_health", return_value={"available": False}):
        success = upload_to_hdfs(local_path=dummy_csv)
        assert success is False
        fallback_file = config.PROC_DIR / "hdfs_fallback" / "test_data.csv"
        assert fallback_file.exists()


def test_upload_to_hdfs_success_when_online(tmp_path):
    """When HDFS is online and CLI commands succeed, upload_to_hdfs returns True."""
    dummy_csv = tmp_path / "online_test.csv"
    dummy_csv.write_text("CustomerId,Exited\n10,1\n")

    mock_run = MagicMock()
    mock_run.returncode = 0
    mock_run.stdout = "Found 1 items\n-rw-r--r-- 1 vikas supergroup 100 /banking_data/online_test.csv"

    with patch("src.hdfs_utils.check_hdfs_health", return_value={"available": True}):
        with patch("src.hdfs_utils.run_hdfs_cmd", return_value=mock_run):
            success = upload_to_hdfs(local_path=dummy_csv)
            assert success is True
