#!/usr/bin/env python
"""End-to-end demo of the pydantic dataset-spec feature set.

Run from the coffea repo root (needs the sample ROOT files under tests/samples)::

    python examples/dataset_tools_demo.py

It exercises, on a real preprocessed ``DataGroupSpec``:

* rich / Jupyter display of specs,
* ServiceX conversions (dict + reverse from a ``deliver()`` result),
* ROOT RDataFrame ``FromSpec`` JSON round-trip,
* universal_pathlib storage-options + open/close file handles,
* CMS ``ParameterSets`` provenance decoding (root-file introspection),
* the explorer's diff / column-filter / what-if / provenance logic,

and points at the interactive ``.explore()`` Textual UI (``pip install coffea[tui]``).
"""

from __future__ import annotations

import json

from coffea.dataset_tools import DataGroupSpec, from_servicex, preprocess
from coffea.dataset_tools._explore import (
    cms_provenance,
    diff_specs,
    filter_columns,
    is_empty_diff,
    provenance,
    whatif,
)
from coffea.util import coffea_console, describe_root_file, extract_cms_provenance

DY = "tests/samples/nano_dy.root"
DIMUON = "tests/samples/nano_dimuon.root"
PFNANO = "tests/samples/pfnano.root"  # a real CMS NanoAOD with a ParameterSets tree


def rule(title: str) -> None:
    coffea_console.rule(f"[bold]{title}")


def main() -> None:
    # ------------------------------------------------------------------ #
    # 0. ROOT file introspection (no event data read)
    # ------------------------------------------------------------------ #
    rule("describe_root_file")
    for name, info in describe_root_file(DY).items():
        nbranches = len(info["branches"] or [])
        coffea_console.print(
            f"  {name}: {info['classname']} "
            f"entries={info['num_entries']} branches={nbranches}"
        )

    # ------------------------------------------------------------------ #
    # 1. Build + preprocess a DataGroupSpec, then display it
    # ------------------------------------------------------------------ #
    rule("preprocess -> DataGroupSpec (rich display)")
    fileset = DataGroupSpec(
        {
            "DYJets": {"files": {DY: "Events"}, "metadata": {"xsec": 6077.22}},
            "DoubleMuon": {"files": {DIMUON: "Events"}, "metadata": {"is_data": True}},
        }
    )
    # backend="iterative" is the dask-free preprocessing path (also handles RNTuple)
    available, allfiles = preprocess(
        fileset, step_size=20_000, save_form=True, backend="iterative"
    )
    coffea_console.print(available)  # __rich__ tree with badge / sparkline / columns

    # ------------------------------------------------------------------ #
    # 2. RDataFrame FromSpec JSON round-trip
    # ------------------------------------------------------------------ #
    rule("RDataFrame RDatasetSpec round-trip")
    rdf_json = available.to_rdf_spec_json()
    coffea_console.print(json.loads(rdf_json)["samples"]["DYJets"])
    # the RDF spec carries files/trees/metadata (not steps/uuid/form), so it
    # round-trips through its own projection rather than to the preprocessed spec
    assert (
        DataGroupSpec.from_rdf_spec(rdf_json).to_rdf_spec() == available.to_rdf_spec()
    )
    coffea_console.print("[green]round-trip through RDataFrame spec is stable[/green]")

    # ------------------------------------------------------------------ #
    # 3. ServiceX conversions (dict form + reverse from a deliver() result)
    # ------------------------------------------------------------------ #
    rule("ServiceX conversions")
    did_group = DataGroupSpec(
        {"DYJets": {"files": {"root://x//in.root": "Events"}, "did": "mc:dy"}}
    )
    coffea_console.print("to_servicex_dict:", did_group.to_servicex_dict())
    delivered = {"DYJets": [DY], "DoubleMuon": [DIMUON]}  # a servicex.deliver() result
    reconstructed = from_servicex(delivered, object_path="Events")
    coffea_console.print(
        "from_servicex -> datasets:", list(reconstructed), "(ready to preprocess)"
    )

    # ------------------------------------------------------------------ #
    # 4. upath storage-options + open/close handles
    # ------------------------------------------------------------------ #
    rule("storage_options (excluded from serialization) + open/close")
    ds = available["DYJets"]
    fs = next(iter(ds.files.values()))
    fs.storage_options = {"token": "SECRET-BEARER-TOKEN"}
    dumped = ds.model_dump_json()
    coffea_console.print("secret token serialized?", "SECRET" in dumped)
    fname = next(iter(ds.files))
    with ds.opened([fname]) as handles:
        head = handles[fname].read(4)
    coffea_console.print(
        f"opened {fname!r}, first bytes: {head!r}; is_open now:", fs.is_open
    )

    # ------------------------------------------------------------------ #
    # 5. CMS ParameterSets provenance decoding
    # ------------------------------------------------------------------ #
    rule("CMS ParameterSets provenance (extract_cms_provenance)")
    # pfnano.root is a real CMS NanoAOD: its ParameterSets tree embeds the whole
    # cmsRun config, from which the global tag / process chain / JEC levels decode.
    prov = extract_cms_provenance(PFNANO)  # curated view
    coffea_console.print("global_tag:", prov["global_tag"])
    coffea_console.print("process chain:", prov["process_names"])
    coffea_console.print("jec_levels:", prov["jec_levels"])
    # regex key search over the full 600+ decoded parameters (the TUI provenance tab)
    hits = cms_provenance(PFNANO, r"globaltag|process_name", detail="full")
    coffea_console.print(f"regex 'globaltag|process_name' -> {len(hits)} decoded keys")

    # ------------------------------------------------------------------ #
    # 6. Explorer logic: diff, column filter, what-if, provenance
    # ------------------------------------------------------------------ #
    rule("explorer: diff available vs all")
    d = diff_specs(available, allfiles)
    coffea_console.print("no differences" if is_empty_diff(d) else d)

    rule("explorer: column filter '^Muon'")
    coffea_console.print([lf.path for lf in filter_columns(ds, "^Muon")][:8])

    rule("explorer: what-if limit_files=1")
    _, summary = whatif(available, max_files=1)
    coffea_console.print(summary)

    rule("explorer: provenance (metadata)")
    coffea_console.print(provenance(ds))

    rule("interactive TUI")
    coffea_console.print(
        "Run [bold]available.explore(other=allfiles)[/bold] for the Textual UI "
        "(requires [bold]pip install coffea[tui][/bold])."
    )


if __name__ == "__main__":
    main()
