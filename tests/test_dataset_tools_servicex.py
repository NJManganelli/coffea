"""Tests for ServiceX <-> pydantic dataset-spec conversions."""

import sys

import pytest

from coffea.dataset_tools import (
    DataGroupSpec,
    DatasetSpec,
    from_servicex,
    to_servicex_dict,
)
from coffea.dataset_tools.servicex import datasetspec_from_delivered


class _GuardListLike:
    """Mimics servicex 3.3.x GuardList: a Sequence, not a list, no .files."""

    def __init__(self, files):
        self._files = list(files)

    def __iter__(self):
        return iter(self._files)


# --------------------------------------------------------------------------- #
# deliver() output -> DataGroupSpec  (dependency-free)
# --------------------------------------------------------------------------- #
def test_from_servicex_root_and_parquet_samples():
    delivered = {
        "dy": ["/cache/dy_0.root", "/cache/dy_1.root"],
        "tt": _GuardListLike(["/cache/tt_0.parquet"]),
    }
    dg = from_servicex(delivered, metadata={"origin": "sx"})
    assert isinstance(dg, DataGroupSpec)
    assert set(dg) == {"dy", "tt"}

    dy = dg["dy"]
    assert dy.format == "root"
    assert len(dy.files) == 2
    # ROOT files get the output tree as object_path
    assert all(fs.object_path == "servicex" for fs in dy.files.values())
    assert dy.metadata == {"origin": "sx"}

    tt = dg["tt"]
    assert tt.format == "parquet"
    # Parquet forbids object_path
    assert all(fs.object_path is None for fs in tt.files.values())


def test_from_servicex_object_path_override_and_dotfiles_attr():
    class _WithFiles:
        files = ["/cache/a.root"]

    dg = from_servicex({"s": _WithFiles()}, object_path="Events")
    assert next(iter(dg["s"].files.values())).object_path == "Events"


def test_from_servicex_rejects_non_mapping_and_empty_sample():
    with pytest.raises(TypeError):
        from_servicex(["/cache/a.root"])
    with pytest.raises(ValueError, match="delivered no files"):
        from_servicex({"empty": []})


def test_datasetspec_from_delivered_accepts_pathlike():
    from pathlib import Path

    ds = datasetspec_from_delivered([Path("/cache/a.root")])
    assert isinstance(ds, DatasetSpec)
    assert list(ds.files) == ["/cache/a.root"]


# --------------------------------------------------------------------------- #
# DataGroupSpec -> ServiceX spec dict  (dependency-free)
# --------------------------------------------------------------------------- #
def _did_group():
    return DataGroupSpec(
        {
            "dy": DatasetSpec(
                files={"root://x//a.root": {"object_path": "Events"}}, did="scope:dy"
            )
        }
    )


def _xrootd_group():
    return DataGroupSpec(
        {
            "dy": DatasetSpec(
                files={
                    "root://x//a.root": {"object_path": "Events"},
                    "root://x//b.root": {"object_path": "Events"},
                }
            )
        }
    )


def _local_group():
    return DataGroupSpec(
        {"dy": DatasetSpec(files={"/data/a.root": {"object_path": "Events"}})}
    )


def test_to_servicex_dict_uses_did_then_xrootd():
    d = to_servicex_dict(_did_group(), codegen="uproot-raw")
    (sample,) = d["Sample"]
    assert sample == {"Name": "dy", "RucioDID": "scope:dy", "Codegen": "uproot-raw"}

    d = to_servicex_dict(_xrootd_group())
    (sample,) = d["Sample"]
    assert sample["Name"] == "dy"
    assert sample["XRootDFiles"] == ["root://x//a.root", "root://x//b.root"]
    assert "RucioDID" not in sample


def test_to_servicex_dict_prefer_did_false_falls_back_to_xrootd():
    # did present but not preferred -> uses the xrootd file URLs instead
    d = to_servicex_dict(_did_group(), prefer_did=False)
    sample = d["Sample"][0]
    assert "RucioDID" not in sample
    assert sample["XRootDFiles"] == ["root://x//a.root"]


def test_to_servicex_dict_local_filelist_raises():
    with pytest.raises(ValueError, match="local FileList cannot be expressed"):
        to_servicex_dict(_local_group())


def test_to_servicex_dict_query_and_general_passthrough():
    d = to_servicex_dict(
        _did_group(),
        query={"type": "UprootRaw"},
        general={"OutputFormat": "root-ttree"},
    )
    assert d["Sample"][0]["Query"] == {"type": "UprootRaw"}
    assert d["General"] == {"OutputFormat": "root-ttree"}


# --------------------------------------------------------------------------- #
# lazy servicex import: dep-free paths work with servicex unavailable
# --------------------------------------------------------------------------- #
def test_class_helpers_are_lazy(monkeypatch):
    # Simulate servicex not installed: `import servicex` then raises ImportError.
    monkeypatch.setitem(sys.modules, "servicex", None)

    # dependency-free conversions still work
    assert from_servicex({"s": ["/cache/a.root"]})["s"].format == "root"
    assert to_servicex_dict(_did_group())["Sample"][0]["RucioDID"] == "scope:dy"

    # class-based conversion needs servicex and surfaces the missing dep
    with pytest.raises((ImportError, TypeError)):
        _did_group().to_servicex()


# --------------------------------------------------------------------------- #
# ServiceX class form (requires servicex installed)
# --------------------------------------------------------------------------- #
def test_to_servicex_spec_builds_real_objects_and_round_trips():
    servicex = pytest.importorskip("servicex")

    # FileList branch (no did preferred)
    spec = _xrootd_group().to_servicex()
    assert isinstance(spec, servicex.ServiceXSpec)
    (sample,) = spec.Sample
    assert sample.Name == "dy"
    assert type(sample.Dataset).__name__ == "FileListDataset"

    # Rucio branch
    ds_sample = _did_group()["dy"].to_servicex_sample(name="dy")
    assert type(ds_sample.Dataset).__name__ == "RucioDatasetIdentifier"

    # the dep-free dict is accepted by servicex (deliver's mapping path)
    servicex.ServiceXSpec(**to_servicex_dict(_did_group()))
    servicex.ServiceXSpec(**to_servicex_dict(_xrootd_group()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
