"""The data folder and the bundled model checksum manifests."""
import hashlib
import sys
from pathlib import Path

from phyloai import models, paths


def test_phyloai_data_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PHYLOAI_DATA", str(tmp_path))
    assert paths.data_dir() == tmp_path
    assert paths.models_dir() == tmp_path / "models"
    assert paths.detector_path() == tmp_path / "ModelDetector.h5"


def test_default_data_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("PHYLOAI_DATA", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    if sys.platform == "darwin":
        assert paths.data_dir() == Path.home() / "Library" / "Application Support" / "phyloai"
    else:
        assert paths.data_dir() == tmp_path / "phyloai"


def test_manifests_are_bundled_and_unchanged():
    for name, sha256, files in (("protein_models.SHA256SUMS", models.MANIFEST_SHA256, 105),
                                ("six_state_la_models.SHA256SUMS",
                                 "edd16cfb8abf3f12026718dfb5afcee40032dde5053d9f2e92cc39b3bd9dcbb1", 19)):
        data = models.manifest_file(name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256
        assert len(data.decode().splitlines()) == files


def test_download_defaults_follow_data_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("PHYLOAI_DATA", str(tmp_path))
    args = models.build_parser().parse_args([])
    assert args.models == tmp_path / "models"
    assert args.detector == tmp_path / "ModelDetector.h5"
    assert args.family == "all" and not args.no_detector


def test_existing_detector_is_not_overwritten(tmp_path):
    target = tmp_path / "ModelDetector.h5"
    target.write_bytes(b"something else")
    try:
        models.download_detector(target)
    except ValueError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("a different existing file must not be accepted")
    assert target.read_bytes() == b"something else"
