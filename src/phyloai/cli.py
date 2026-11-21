"""The `phyloai` command: `phyloai <command> [options]`.

Each command is a module with `main(argv, prog)`. Modules are imported only when their command
runs, so `phyloai --help` and `phyloai check` start quickly and do not load TensorFlow.
"""
from __future__ import annotations

import importlib
import sys

from phyloai import __version__

# name: (module, function, one-line help)
COMMANDS = {
    "run": ("phyloai.run", "main",
            "Reconstruct a species tree from protein alignments"),
    "download-models": ("phyloai.models", "main",
                        "Download and verify the pretrained networks and ModelDetector"),
    "check": ("phyloai.check", "main",
              "Check the installation: Python packages, external tools and models"),
    "simulate": ("phyloai.simulate", "main",
                 "Simulate training data with PolyMoSim"),
    "train": ("phyloai.train", "main",
              "Train Fisher, six-state or full-pattern networks"),
    "recode-6state": ("phyloai.recode", "main",
                      "Collapse 20-state site pattern frequencies into a six-state recoding"),
    "test-datasets": ("phyloai.benchmark", "main",
                      "Simulate test alignments and evaluate networks (and IQ-TREE)"),

}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: phyloai <command> [options]", "",
             "PhyloAI reconstructs phylogenetic trees from protein alignments with neural networks.", "",
             "commands:"]
    lines += [f"  {name:<{width}}  {text}" for name, (_, _, text) in COMMANDS.items()]
    lines += ["", "Run `phyloai <command> --help` for the options of a command.",
              "Other options: -h/--help, -V/--version"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"phyloai {__version__}")
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(usage(), file=sys.stderr)
        print(f"\nphyloai: unknown command '{name}'", file=sys.stderr)
        return 2
    module, function, _ = COMMANDS[name]
    result = getattr(importlib.import_module(module), function)(rest, prog=f"phyloai {name}")
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
