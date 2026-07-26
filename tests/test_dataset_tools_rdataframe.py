"""Tests for RDataFrame RDatasetSpec <-> pydantic dataset-spec conversions."""

import json

import pytest

from coffea.dataset_tools import (
    DataGroupSpec,
    DatasetSpec,
    from_rdf_spec,
    to_rdf_spec,
    to_rdf_spec_json,
)


def _root_ds(files_trees, metadata=None):
    files = {f: {"object_path": t} for f, t in files_trees}
    return DatasetSpec(files=files, metadata=metadata or {})


# --------------------------------------------------------------------------- #
# DataGroupSpec -> RDataFrame spec
# --------------------------------------------------------------------------- #
def test_to_rdf_spec_shared_tree_collapses_to_single_entry():
    dg = DataGroupSpec(
        {"dy": _root_ds([("a.root", "Events"), ("b.root", "Events")], {"xsec": 2.0})}
    )
    spec = to_rdf_spec(dg)
    sample = spec["samples"]["dy"]
    assert sample["trees"] == ["Events"]  # one shared tree, not repeated
    assert sample["files"] == ["a.root", "b.root"]
    assert sample["metadata"] == {"xsec": 2.0}


def test_to_rdf_spec_per_file_trees_when_mixed():
    dg = DataGroupSpec({"m": _root_ds([("a.root", "t1"), ("b.root", "t2")])})
    sample = to_rdf_spec(dg)["samples"]["m"]
    assert sample["trees"] == ["t1", "t2"]  # per-file trees
    assert "metadata" not in sample  # empty metadata omitted


def test_to_rdf_spec_rejects_parquet():
    dg = DataGroupSpec({"pq": DatasetSpec(files={"a.parquet": None})})
    with pytest.raises(ValueError, match="ROOT TTrees only"):
        to_rdf_spec(dg)


def test_to_rdf_spec_json_writes_file(tmp_path):
    dg = DataGroupSpec({"dy": _root_ds([("a.root", "Events")])})
    out = tmp_path / "spec.json"
    text = to_rdf_spec_json(dg, out)
    assert json.loads(text) == json.loads(out.read_text())
    assert json.loads(text)["samples"]["dy"]["trees"] == ["Events"]


# --------------------------------------------------------------------------- #
# RDataFrame spec -> DataGroupSpec
# --------------------------------------------------------------------------- #
def test_from_rdf_spec_expands_shared_tree():
    spec = {
        "samples": {
            "dy": {
                "trees": ["Events"],
                "files": ["a.root", "b.root"],
                "metadata": {"xsec": 2.0},
            }
        }
    }
    dg = from_rdf_spec(spec)
    assert isinstance(dg, DataGroupSpec)
    dy = dg["dy"]
    assert dy.format == "root"
    assert [fs.object_path for fs in dy.files.values()] == ["Events", "Events"]
    assert dy.metadata == {"xsec": 2.0}


def test_from_rdf_spec_accepts_json_string_and_path(tmp_path):
    spec = {"samples": {"dy": {"trees": "Events", "files": ["a.root"]}}}
    # single tree name given as a bare string is accepted
    dg = from_rdf_spec(json.dumps(spec))
    assert next(iter(dg["dy"].files.values())).object_path == "Events"

    p = tmp_path / "s.json"
    p.write_text(json.dumps(spec))
    assert from_rdf_spec(p)["dy"].format == "root"
    assert from_rdf_spec(str(p))["dy"].format == "root"


def test_from_rdf_spec_tree_length_mismatch_raises():
    spec = {"samples": {"m": {"trees": ["t1", "t2"], "files": ["a.root"]}}}
    with pytest.raises(ValueError, match="must have length 1 or len\\(files\\)"):
        from_rdf_spec(spec)


def test_from_rdf_spec_friends_warn_and_ignored():
    spec = {
        "samples": {"dy": {"trees": ["Events"], "files": ["a.root"]}},
        "friends": {"f": {"trees": ["Friends"], "files": ["fr.root"]}},
    }
    with pytest.warns(UserWarning, match="friend trees"):
        dg = from_rdf_spec(spec)
    assert set(dg) == {"dy"}  # friends dropped


def test_from_rdf_spec_missing_samples_raises():
    with pytest.raises(ValueError, match="'samples' mapping"):
        from_rdf_spec({"metadata": {}})


# --------------------------------------------------------------------------- #
# round-trip + model methods
# --------------------------------------------------------------------------- #
def test_round_trip_via_model_methods():
    dg = DataGroupSpec(
        {
            "dy": _root_ds([("a.root", "Events"), ("b.root", "Events")], {"xsec": 2.0}),
            "mix": _root_ds([("c.root", "t1"), ("d.root", "t2")]),
        }
    )
    restored = DataGroupSpec.from_rdf_spec(dg.to_rdf_spec_json())
    assert restored == dg
    assert restored.to_rdf_spec() == dg.to_rdf_spec()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
