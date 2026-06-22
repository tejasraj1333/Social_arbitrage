"""CLI smoke test."""

from __future__ import annotations

from sam.cli import main


def test_cli_check_returns_zero() -> None:
    assert main(["check"]) == 0
