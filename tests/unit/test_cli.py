"""Tests for CLI."""

import pytest
from unittest.mock import patch, MagicMock

from lumen.interface.cli import main


class TestCLI:
    def test_help_exits_cleanly(self):
        """lumen --help should not crash."""
        with patch("sys.argv", ["lumen"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_cost_missing_project(self):
        """lumen cost with nonexistent project → error."""
        with patch("sys.argv", ["lumen", "cost", "--project", "/nonexistent/path"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_validate_missing_project(self):
        """lumen validate with nonexistent output → error."""
        with patch("sys.argv", ["lumen", "validate", "--project", "/nonexistent/path"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_run_subcommand_exists(self):
        """Run subcommand is registered."""
        import argparse
        from lumen.interface.cli import main as cli_main
        # Just verify it parses without error
        with patch("sys.argv", ["lumen", "run", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 0
