"""Pure (non-UI) logic backing the dataset-spec explorer.

Everything here is plain Python -- no ``textual`` import -- so it is unit-testable
and usable programmatically. The interactive Textual app in ``_explore_app`` is a
thin presentation layer over these functions and is launched via :func:`explore`
(which requires the optional ``textual`` dependency, i.e. ``coffea[tui]``).

Four capabilities:

* :func:`diff_specs` -- compare two filesets (e.g. ``preprocess``'s *available*
  vs *all*): datasets/files added or removed, changed step and entry counts.
* :func:`filter_columns` -- regex filter over a dataset's form columns.
* :func:`whatif` -- apply ``filter_files``/``limit_files``/``limit_steps`` and
  report the resulting counts (files, entries, partitions); the returned spec
  is exportable to JSON.
* :func:`provenance` / :func:`cms_provenance` / :func:`search_keys` -- inspect a
  dataset's metadata (and, when available, a ROOT file's CMS ``ParameterSets``
  provenance) with a regex key search.
"""

from __future__ import annotations

import re
from typing import Any

from coffea.dataset_tools._display import Leaf, column_leaves
from coffea.dataset_tools.filespec import DataGroupSpec, DatasetSpec


def _as_group(spec: Any) -> DataGroupSpec:
    """Coerce a DatasetSpec/DataGroupSpec to a DataGroupSpec view."""
    if isinstance(spec, DataGroupSpec):
        return spec
    if isinstance(spec, DatasetSpec):
        return DataGroupSpec({"": spec})
    raise TypeError(f"expected a DatasetSpec or DataGroupSpec, got {type(spec)}")


# --------------------------------------------------------------------------- #
# regex key search over nested blobs (metadata / provenance)
# --------------------------------------------------------------------------- #
def search_keys(
    blob: Any,
    pattern: str | None = None,
    *,
    search_values: bool = False,
    flags: int = re.IGNORECASE,
) -> list[tuple[str, Any]]:
    """Return ``(dotted_path, value)`` leaves of *blob* whose key path (or value,
    when *search_values*) matches *pattern*; ``None`` returns every leaf."""
    regex = re.compile(pattern, flags) if pattern else None
    out: list[tuple[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        else:
            if regex is None:
                out.append((path, node))
            elif regex.search(str(node) if search_values else path):
                out.append((path, node))

    walk(blob, "")
    return out


# --------------------------------------------------------------------------- #
# column explorer
# --------------------------------------------------------------------------- #
def _form_of(spec_or_form: Any) -> Any:
    return spec_or_form.form if isinstance(spec_or_form, DatasetSpec) else spec_or_form


def filter_columns(
    spec_or_form: Any, pattern: str = "", *, search_doc: bool = True
) -> list[Leaf]:
    """Regex-filter a dataset's form columns by name (and title when *search_doc*).

    Accepts a DatasetSpec or a form (awkward Form / dict / json str). An empty
    *pattern* returns all leaves.
    """
    leaves = column_leaves(_form_of(spec_or_form))
    if not pattern:
        return leaves
    regex = re.compile(pattern, re.IGNORECASE)
    return [
        lf
        for lf in leaves
        if regex.search(lf.path) or (search_doc and lf.doc and regex.search(lf.doc))
    ]


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def diff_datasets(a: DatasetSpec, b: DatasetSpec) -> dict[str, Any]:
    """Structured differences between two DatasetSpecs (empty dict if identical)."""
    fa, fb = a.files, b.files
    keys_a, keys_b = set(fa), set(fb)
    out: dict[str, Any] = {}
    if only_a := sorted(keys_a - keys_b):
        out["files_only_in_a"] = only_a
    if only_b := sorted(keys_b - keys_a):
        out["files_only_in_b"] = only_b
    steps_changed = {
        k: (fa[k].steps, fb[k].steps)
        for k in sorted(keys_a & keys_b)
        if fa[k].steps != fb[k].steps
    }
    if steps_changed:
        out["steps_changed"] = steps_changed
    if a.num_entries != b.num_entries:
        out["num_entries"] = (a.num_entries, b.num_entries)
    if a.num_selected_entries != b.num_selected_entries:
        out["num_selected_entries"] = (a.num_selected_entries, b.num_selected_entries)
    return out


def diff_specs(a: Any, b: Any) -> dict[str, Any]:
    """Compare two filesets (DatasetSpec or DataGroupSpec).

    Returns datasets present in only one side and, for shared datasets, the
    per-dataset :func:`diff_datasets` differences. The canonical use is diffing
    ``preprocess``'s *available* against *all*.
    """
    a, b = _as_group(a), _as_group(b)
    names_a, names_b = set(a), set(b)
    result: dict[str, Any] = {
        "datasets_only_in_a": sorted(names_a - names_b),
        "datasets_only_in_b": sorted(names_b - names_a),
        "changed": {},
    }
    for name in sorted(names_a & names_b):
        d = diff_datasets(a[name], b[name])
        if d:
            result["changed"][name] = d
    return result


def is_empty_diff(diff: dict[str, Any]) -> bool:
    """True when :func:`diff_specs` found no differences."""
    return not (
        diff["datasets_only_in_a"] or diff["datasets_only_in_b"] or diff["changed"]
    )


# --------------------------------------------------------------------------- #
# what-if slicing
# --------------------------------------------------------------------------- #
def summarize(spec: Any) -> dict[str, Any]:
    """Count datasets, files, entries, selected entries and step-partitions."""
    group = _as_group(spec)
    n_files = sum(len(ds.files) for ds in group.values())
    partitions = sum(
        len(fs.steps or []) for ds in group.values() for fs in ds.files.values()
    )
    return {
        "datasets": len(group),
        "files": n_files,
        "num_entries": spec.num_entries,
        "num_selected_entries": spec.num_selected_entries,
        "partitions": partitions,
    }


def whatif(
    spec: Any,
    *,
    filter_name: str | None = None,
    max_files: int | None = None,
    max_steps: int | None = None,
    per_file: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Apply filter/limit operations and return ``(result_spec, summary)``.

    Operations are applied in order: ``filter_files`` (regex on file name),
    ``limit_files``, then ``limit_steps``. *result_spec* is the same type as
    *spec* and can be serialized with ``model_dump_json()``.
    """
    result = spec
    if filter_name is not None:
        result = result.filter_files(filter_name=filter_name)
    if max_files is not None:
        result = result.limit_files(max_files)
    if max_steps is not None:
        result = result.limit_steps(max_steps, per_file=per_file)
    return result, summarize(result)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def provenance(spec: DatasetSpec, pattern: str | None = None) -> dict[str, Any]:
    """A dataset's metadata (incl. any ``column_join`` lineage), regex-filtered.

    With *pattern*, returns the flattened ``{dotted_path: value}`` matches;
    without it, the raw metadata dict.
    """
    metadata = dict(getattr(spec, "metadata", {}) or {})
    if pattern:
        return dict(search_keys(metadata, pattern))
    return metadata


def cms_provenance(
    filename: str, pattern: str | None = None, detail: str = "full"
) -> dict[str, Any]:
    """CMS ``ParameterSets`` provenance for a ROOT *filename*, regex-filtered.

    Delegates to ``coffea.util.extract_cms_provenance`` (the ROOT-file
    introspection feature); raises if that helper is unavailable. With *pattern*,
    only matching keys are returned.
    """
    from coffea import util

    extractor = getattr(util, "extract_cms_provenance", None)
    if extractor is None:
        raise RuntimeError(
            "coffea.util.extract_cms_provenance is unavailable; it ships with the "
            "ROOT-file-introspection feature"
        )
    prov = extractor(filename, detail=detail) or {}
    if pattern:
        return dict(search_keys(prov, pattern))
    return prov


# --------------------------------------------------------------------------- #
# launcher
# --------------------------------------------------------------------------- #
def explore(spec: Any, other: Any = None) -> None:
    """Launch the interactive Textual explorer for *spec*.

    Pass *other* to open directly in diff mode against a second fileset. Requires
    the optional ``textual`` dependency (``pip install coffea[tui]``).
    """
    try:
        from coffea.dataset_tools._explore_app import DatasetExplorerApp
    except ModuleNotFoundError as exc:  # pragma: no cover - requires textual absent
        raise ModuleNotFoundError(
            "The dataset explorer TUI requires 'textual'; install it with "
            "'pip install coffea[tui]'."
        ) from exc
    DatasetExplorerApp(spec, other=other).run()
