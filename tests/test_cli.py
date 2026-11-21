"""The `phyloai` command and its subcommands."""
import subprocess
import sys

import pytest

from phyloai import __version__
from phyloai.cli import COMMANDS, main


def phyloai(*arguments):
    return subprocess.run([sys.executable, "-m", "phyloai", *arguments], capture_output=True, text=True, check=False)


def test_help_lists_all_commands(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for name in COMMANDS:
        assert name in output


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"phyloai {__version__}"


def test_unknown_command(capsys):
    assert main(["no-such-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("name", list(COMMANDS))
def test_command_help(name):
    result = phyloai(name, "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"usage: phyloai {name}")


def test_python_m_phyloai():
    result = phyloai("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == f"phyloai {__version__}"
