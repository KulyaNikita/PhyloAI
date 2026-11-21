"""Input checks of `phyloai run` and the six-state recoding."""
from types import SimpleNamespace

import numpy as np
import pytest

from phyloai import recode, run


@pytest.mark.parametrize("version", ["1.8.0", "1.8.3", "1.9.0"])
def test_supported_qpc_versions(version):
    run.check_qpc(SimpleNamespace(__version__=version, AA_PATTERN_ORDER=run.AA_ORDER, __file__="qpc.so"))


@pytest.mark.parametrize("version,order", [("1.7.1", run.AA_ORDER), ("2.0.0", run.AA_ORDER),
                                           ("1.8.0", "ARNDCQEGHILKMFPSTWYV"), ("", run.AA_ORDER)])
def test_unsupported_qpc(version, order):
    with pytest.raises(RuntimeError, match="QPC >=1.8,<2"):
        run.check_qpc(SimpleNamespace(__version__=version, AA_PATTERN_ORDER=order, __file__="qpc.so"))


def test_installed_qpc_is_supported():
    qpc = pytest.importorskip("QuartetPatternCounterModule")
    run.check_qpc(qpc)


def test_read_alignment(tmp_path):
    pytest.importorskip("Bio")
    good = tmp_path / "good.fasta"
    good.write_text(">a\nACDE\n>b\nACDF\n>c\nACDG\n>d\nAC-H\n")
    names, sequences = run.read_alignment(good, "fasta")
    assert names == ["a", "b", "c", "d"] and sequences[3] == "AC-H"
    three = tmp_path / "three.fasta"
    three.write_text(">a\nACDE\n>b\nACDF\n>c\nACDG\n")
    with pytest.raises(ValueError, match=">=4 distinct"):
        run.read_alignment(three, "fasta")


def test_alignment_paths(tmp_path):
    (tmp_path / "gene1.fasta").write_text("")
    (tmp_path / "gene2.fas").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert [p.name for p in run.alignment_paths(tmp_path)] == ["gene1.fasta", "gene2.fas"]


def test_recode_batch_sums_to_one():
    frequencies = np.full((2, 160000), 1 / 160000, dtype=np.float64)
    for scheme in recode.SCHEMES:
        recoded = recode.recode_batch(frequencies.copy(), recode.group_indices(scheme))
        assert recoded.shape == (2, 1296)
        assert np.allclose(recoded.sum(axis=1), 1)
