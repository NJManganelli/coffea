"""Tests for the pure (non-UI) dataset-explorer logic."""

import json

import awkward as ak
import pytest

from coffea.dataset_tools._explore import (
    diff_specs,
    filter_columns,
    is_empty_diff,
    provenance,
    search_keys,
    summarize,
    whatif,
)
from coffea.dataset_tools.filespec import DataGroupSpec, DatasetSpec


def _form_json():
    arr = ak.Array(
        [{"Jet_pt": [1.0], "Jet_eta": [0.1], "Muon_pt": [2.0], "nJet": 1, "run": 1}]
    )
    return arr.layout.form.to_json()


def _ds(files, metadata=None, form=None):
    kw = {"files": files, "metadata": metadata or {}}
    if form is not None:
        kw["form"] = form
    return DatasetSpec(**kw)


def _preprocessed():
    return _ds(
        {
            "a.root": {
                "object_path": "Events",
                "num_entries": 1000,
                "uuid": "ua",
                "steps": [[0, 500], [500, 1000]],
            },
            "b.root": {
                "object_path": "Events",
                "num_entries": 300,
                "uuid": "ub",
                "steps": [[0, 300]],
            },
        },
        metadata={"xsec": 6.0, "column_join": {"engine": "trino", "keys": ["event"]}},
        form=_form_json(),
    )


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def test_diff_specs_detects_dataset_and_file_and_step_changes():
    full = DataGroupSpec({"dy": _preprocessed(), "extra": _preprocessed()})
    # available: 'extra' missing, 'dy' lost file b and shrank a's steps
    available = DataGroupSpec(
        {
            "dy": _ds(
                {
                    "a.root": {
                        "object_path": "Events",
                        "num_entries": 1000,
                        "uuid": "ua",
                        "steps": [[0, 500]],
                    }
                }
            )
        }
    )
    d = diff_specs(available, full)
    assert d["datasets_only_in_b"] == ["extra"]
    assert d["datasets_only_in_a"] == []
    change = d["changed"]["dy"]
    assert change["files_only_in_b"] == ["b.root"]
    assert "a.root" in change["steps_changed"]
    assert change["steps_changed"]["a.root"] == ([[0, 500]], [[0, 500], [500, 1000]])
    assert change["num_entries"] == (1000, 1300)  # file-count sum (a only vs a+b)
    assert change["num_selected_entries"] == (500, 1300)  # from steps
    assert not is_empty_diff(d)


def test_diff_specs_identical_is_empty():
    dg = DataGroupSpec({"dy": _preprocessed()})
    d = diff_specs(dg, DataGroupSpec({"dy": _preprocessed()}))
    assert is_empty_diff(d)
    assert d["changed"] == {}


def test_diff_accepts_single_datasetspec():
    d = diff_specs(_preprocessed(), _preprocessed())
    assert is_empty_diff(d)


# --------------------------------------------------------------------------- #
# column filter
# --------------------------------------------------------------------------- #
def test_filter_columns_regex_over_names():
    ds = _preprocessed()
    # substring match also catches nJet
    assert {lf.path for lf in filter_columns(ds, "Jet")} == {
        "Jet_pt",
        "Jet_eta",
        "nJet",
    }
    # anchored pattern restricts to the Jet collection
    assert {lf.path for lf in filter_columns(ds, "^Jet")} == {"Jet_pt", "Jet_eta"}
    assert len(filter_columns(ds, "")) == 5  # empty pattern -> all leaves


def test_filter_columns_matches_doc_title():
    form = {
        "class": "RecordArray",
        "fields": ["x"],
        "contents": [
            {
                "class": "NumpyArray",
                "primitive": "float32",
                "parameters": {"__doc__": "transverse momentum"},
            }
        ],
        "parameters": {},
    }
    assert [lf.path for lf in filter_columns(form, "momentum")] == ["x"]
    assert filter_columns(form, "momentum", search_doc=False) == []


# --------------------------------------------------------------------------- #
# what-if
# --------------------------------------------------------------------------- #
def test_summarize_counts_partitions():
    s = summarize(DataGroupSpec({"dy": _preprocessed()}))
    assert s == {
        "datasets": 1,
        "files": 2,
        "num_entries": 1300,
        "num_selected_entries": 1300,
        "partitions": 3,  # 2 steps in a.root + 1 in b.root
    }


def test_whatif_limit_files_and_steps_updates_counts_and_exports():
    dg = DataGroupSpec({"dy": _preprocessed()})
    result, summary = whatif(dg, max_files=1)
    assert summary["files"] == 1
    assert summary["num_entries"] == 1000  # only a.root remains
    # result is a live spec, serializable
    assert json.loads(result.model_dump_json())

    _, summary2 = whatif(dg, max_steps=1)
    assert summary2["partitions"] == 1  # cumulative single step across the group


def test_whatif_filter_name():
    dg = DataGroupSpec({"dy": _preprocessed()})
    _, summary = whatif(dg, filter_name="b.root")
    assert summary["files"] == 1
    assert summary["num_entries"] == 300


# --------------------------------------------------------------------------- #
# provenance / key search
# --------------------------------------------------------------------------- #
def test_search_keys_paths_and_values():
    blob = {"column_join": {"engine": "trino", "keys": ["event", "run"]}}
    paths = dict(search_keys(blob, "engine"))
    assert paths == {"column_join.engine": "trino"}
    # value search
    vals = search_keys(blob, "trino", search_values=True)
    assert ("column_join.engine", "trino") in vals
    # list indices are addressable
    all_paths = dict(search_keys(blob))
    assert all_paths["column_join.keys[0]"] == "event"


def test_provenance_returns_metadata_and_filters():
    ds = _preprocessed()
    assert provenance(ds)["xsec"] == 6.0
    filtered = provenance(ds, "column_join")
    assert filtered == {
        "column_join.engine": "trino",
        "column_join.keys[0]": "event",
    }


# --------------------------------------------------------------------------- #
# model hook + TUI launcher guard
# --------------------------------------------------------------------------- #
def test_explore_method_requires_textual():
    ds = _preprocessed()
    try:
        import textual  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError, match="coffea\\[tui\\]"):
            ds.explore()


def test_textual_app_pilot():
    """Drive the real Textual app headlessly (skipped without the coffea[tui] extra)."""
    pytest.importorskip("textual")
    import asyncio

    from coffea.dataset_tools._explore_app import DatasetExplorerApp

    full = DataGroupSpec({"dy": _preprocessed(), "tt": _preprocessed()})
    available = DataGroupSpec({"dy": _preprocessed()})

    async def drive():
        app = DatasetExplorerApp(full, other=available)
        async with app.run_test() as pilot:
            app.query_one("#column-filter").value = "^Jet"
            await pilot.pause()
            assert app.query_one("#column-table").row_count == 2  # Jet_pt, Jet_eta
            app.query_one("#wi-maxfiles").value = "1"
            await pilot.pause()
            assert len(app._whatif_result.files) == 1
            app.query_one("#prov-filter").value = "column_join"
            await pilot.pause()
            assert app.query_one("#prov-table").row_count >= 1
            app.query_one("#dataset-select").value = "tt"
            await pilot.pause()
            assert app._current == "tt"

    asyncio.run(drive())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
