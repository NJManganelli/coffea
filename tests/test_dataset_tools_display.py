"""Tests for the rich / Jupyter display of the pydantic dataset specs."""

import awkward as ak
import pytest
from rich.console import Console
from rich.tree import Tree

from coffea.dataset_tools._display import (
    Leaf,
    column_leaves,
    datagroup_html,
    datagroup_tree,
    dataset_html,
    dataset_tree,
)
from coffea.dataset_tools.filespec import DataGroupSpec, DatasetSpec


def _form_json():
    arr = ak.Array(
        [
            {
                "Jet_pt": [1.0],
                "Jet_eta": [0.1],
                "Jet_phi": [0.2],
                "nJet": 1,
                "run": 1,
                "event": 2,
            }
        ]
    )
    return arr.layout.form.to_json()


def _preprocessed_dataset():
    return DatasetSpec(
        **{
            "files": {
                "root://x//a.root": {
                    "object_path": "Events",
                    "num_entries": 1000,
                    "uuid": "u-a",
                    "steps": [[0, 500], [500, 1000]],
                },
                "root://x//b.root": {
                    "object_path": "Events",
                    "num_entries": 300,
                    "uuid": "u-b",
                    "steps": [[0, 300]],
                },
            },
            "metadata": {"xsec": 6077.22, "is_data": False},
            "form": _form_json(),
        }
    )


def _partial_dataset():
    # No steps/num_entries/uuid -> not concrete, no form.
    return DatasetSpec(**{"files": {"root://x//c.root": {"object_path": "Events"}}})


def _render_text(renderable) -> str:
    con = Console(record=True, width=120)
    con.print(renderable)
    return con.export_text()


# --------------------------------------------------------------------------- #
# column_leaves: dtype + title extraction and grouping input
# --------------------------------------------------------------------------- #
def test_column_leaves_extracts_dtype_and_doc():
    form = {
        "class": "RecordArray",
        "fields": ["Jet_pt", "nJet"],
        "contents": [
            {
                "class": "ListOffsetArray",
                "offsets": "i64",
                "content": {
                    "class": "NumpyArray",
                    "primitive": "float32",
                    "parameters": {},
                },
                "parameters": {"__doc__": "transverse momentum"},
            },
            {"class": "NumpyArray", "primitive": "int32", "parameters": {}},
        ],
        "parameters": {},
    }
    leaves = {leaf.path: leaf for leaf in column_leaves(form)}
    assert set(leaves) == {"Jet_pt", "nJet"}
    # dtype pulled from the innermost NumpyArray, doc inherited from the list wrapper
    assert leaves["Jet_pt"] == Leaf("Jet_pt", "float32", "transverse momentum")
    assert leaves["nJet"].dtype == "int32"
    assert leaves["nJet"].doc is None


def test_column_leaves_none_and_string_forms_agree():
    fj = _form_json()
    assert column_leaves(None) == []
    from_str = {leaf.path for leaf in column_leaves(fj)}
    from_form = {leaf.path for leaf in column_leaves(ak.forms.from_json(fj))}
    assert from_str == from_form
    assert {"Jet_pt", "Jet_eta", "nJet", "run"} <= from_str


# --------------------------------------------------------------------------- #
# rich tree rendering
# --------------------------------------------------------------------------- #
def test_dataset_tree_is_tree_with_status_and_columns():
    tree = dataset_tree(_preprocessed_dataset(), name="DY")
    assert isinstance(tree, Tree)
    text = _render_text(tree)
    assert "DY" in text
    assert "preprocessed" in text  # completeness badge
    assert "1,300 entries" in text  # summed num_entries
    assert "Jet" in text and "Muon" not in text  # grouped columns
    assert "xsec" in text  # metadata surfaced


def test_partial_dataset_badge_distinguishes_from_preprocessed():
    full = _render_text(dataset_tree(_preprocessed_dataset()))
    partial = _render_text(dataset_tree(_partial_dataset()))
    assert "✓ preprocessed" in full
    assert "0/1 preprocessed" in partial  # not fully known
    assert "✓ preprocessed" not in partial


def test_datagroup_tree_lists_each_dataset():
    dg = DataGroupSpec({"a": _preprocessed_dataset(), "b": _partial_dataset()})
    text = _render_text(datagroup_tree(dg))
    assert "DataGroupSpec" in text
    assert "2 datasets" in text
    assert "a" in text and "b" in text


# --------------------------------------------------------------------------- #
# HTML rendering (collapsible <details>, escaping)
# --------------------------------------------------------------------------- #
def test_dataset_html_has_collapsibles_and_dtype():
    html = dataset_html(_preprocessed_dataset(), name="DY")
    assert "<details" in html and "<summary>" in html
    assert "preprocessed" in html
    assert "Jet" in html
    assert "float64" in html  # dtype annotation present in column view


def test_dataset_html_escapes_metadata():
    ds = DatasetSpec(
        **{
            "files": {
                "root://x//a.root": {
                    "object_path": "Events",
                    "num_entries": 10,
                    "uuid": "u",
                    "steps": [[0, 10]],
                }
            },
            "metadata": {"note": "<b>danger</b>"},
        }
    )
    html = dataset_html(ds)
    assert "&lt;b&gt;danger&lt;/b&gt;" in html  # payload escaped
    assert "<b>danger</b>" not in html  # never emitted raw


def test_datagroup_html_wraps_datasets():
    dg = DataGroupSpec({"a": _preprocessed_dataset(), "b": _partial_dataset()})
    html = datagroup_html(dg)
    assert html.count("<details") >= 3  # group + one per dataset (+ nested)
    assert "DataGroupSpec" in html and "2 datasets" in html


# --------------------------------------------------------------------------- #
# model auto-hooks
# --------------------------------------------------------------------------- #
def test_models_autohook_rich_and_html():
    ds = _preprocessed_dataset()
    assert isinstance(ds.__rich__(), Tree)
    assert ds._repr_html_().startswith("<div")

    dg = DataGroupSpec({"a": ds})
    assert isinstance(dg.__rich__(), Tree)
    assert "DataGroupSpec" in dg._repr_html_()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
