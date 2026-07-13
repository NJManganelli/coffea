"""Conversions between the pydantic dataset specs and ROOT RDataFrame's dataset
spec JSON (``ROOT.RDF.Experimental.FromSpec``).

The RDataFrame spec is a small JSON document::

    {"samples": {"<name>": {"trees": [...], "files": [...],
                            "metadata": {...}}}}

which maps almost 1:1 onto a :class:`~coffea.dataset_tools.filespec.DataGroupSpec`:
each sample is a dataset, ``trees`` is the per-sample TTree name(s) (coffea's
``object_path``), and per-sample ``metadata`` is exactly the
``DefinePerSample`` / ``RSampleInfo`` provenance use case.

These converters are pure JSON -- no PyROOT dependency is needed to *produce* a
spec file for ``FromSpec`` to consume.

Limitations
-----------
* RDataFrame reads ROOT TTrees only, so Parquet datasets cannot be exported.
* ``trees`` in RDataFrame is per *sample*; a coffea dataset that mixes TTree
  names across its files is exported as a per-file tree list (``len(trees) ==
  len(files)``) and rejected only if that would be ambiguous.
* RDataFrame *friend trees* and global (spec-level) metadata have no coffea
  analogue; friends present on read are ignored with a warning.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from coffea.dataset_tools.filespec import DataGroupSpec, DatasetSpec


# --------------------------------------------------------------------------- #
# DataGroupSpec -> RDataFrame spec
# --------------------------------------------------------------------------- #
def _sample_dict(spec: DatasetSpec) -> dict[str, Any]:
    if spec.format != "root":
        raise ValueError(
            f"RDataFrame reads ROOT TTrees only; cannot export a {spec.format!r} dataset"
        )
    files = [str(k) for k in spec.files.keys()]
    trees = [fs.object_path for fs in spec.files.values()]
    if any(t is None for t in trees):
        raise ValueError(
            "every file needs an object_path (TTree name) to build an RDataFrame spec"
        )
    unique = list(dict.fromkeys(trees))
    # RDataFrame accepts one tree for all files, or one tree per file.
    sample: dict[str, Any] = {
        "trees": [unique[0]] if len(unique) == 1 else trees,
        "files": files,
    }
    if spec.metadata:
        sample["metadata"] = dict(spec.metadata)
    return sample


def to_rdf_spec(group: DataGroupSpec) -> dict[str, Any]:
    """Convert a DataGroupSpec into an RDataFrame dataset-spec dict.

    The result is suitable for ``json.dump`` and ``ROOT.RDF.Experimental.FromSpec``.
    """
    return {"samples": {name: _sample_dict(spec) for name, spec in group.items()}}


def to_rdf_spec_json(
    group: DataGroupSpec,
    path: str | Path | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize a DataGroupSpec as RDataFrame spec JSON.

    Returns the JSON string; when *path* is given, also writes it there.
    """
    text = json.dumps(to_rdf_spec(group), indent=indent)
    if path is not None:
        Path(path).write_text(text)
    return text


# --------------------------------------------------------------------------- #
# RDataFrame spec -> DataGroupSpec
# --------------------------------------------------------------------------- #
def _load(spec: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        return dict(spec)
    if isinstance(spec, Path):
        return json.loads(spec.read_text())
    if isinstance(spec, str):
        # a JSON document, else a path to one
        try:
            return json.loads(spec)
        except json.JSONDecodeError:
            return json.loads(Path(spec).read_text())
    raise TypeError(
        f"from_rdf_spec expects a mapping, JSON string, or path, got {type(spec)}"
    )


def from_rdf_spec(spec: Mapping[str, Any] | str | Path) -> DataGroupSpec:
    """Convert an RDataFrame dataset spec (dict, JSON string, or path) to a DataGroupSpec.

    Per-sample ``trees`` may be a single tree shared by all files, or one tree
    per file. Sample ``metadata`` is carried onto the DatasetSpec. Friend trees
    are unsupported and ignored with a warning.
    """
    data = _load(spec)
    if "friends" in data:
        warnings.warn(
            "RDataFrame friend trees have no coffea analogue and are ignored",
            stacklevel=2,
        )
    samples = data.get("samples")
    if not isinstance(samples, Mapping):
        raise ValueError("RDataFrame spec must contain a 'samples' mapping")

    group: dict[str, DatasetSpec] = {}
    for name, sample in samples.items():
        files = sample["files"]
        trees = sample["trees"]
        if isinstance(trees, str):
            trees = [trees]
        if len(trees) == 1:
            per_file = trees * len(files)
        elif len(trees) == len(files):
            per_file = list(trees)
        else:
            raise ValueError(
                f"sample {name!r}: 'trees' must have length 1 or len(files) "
                f"({len(files)}), got {len(trees)}"
            )
        files_map = {str(f): {"object_path": t} for f, t in zip(files, per_file)}
        group[name] = DatasetSpec(
            files=files_map, metadata=dict(sample.get("metadata", {}))
        )
    return DataGroupSpec(group)
