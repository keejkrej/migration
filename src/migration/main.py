from __future__ import annotations

import click

from migration import commands  # noqa: F401
from migration.app import app


def main(argv: list[str] | None = None) -> int:
    try:
        result = app(args=argv, prog_name="migration", standalone_mode=False)
    except click.exceptions.BadParameter:
        return 2
    return result or 0
