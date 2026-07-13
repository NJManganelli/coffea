"""Rich / Jupyter display helpers for the pydantic dataset specs.

This module renders :class:`~coffea.dataset_tools.filespec.DatasetSpec` and
:class:`~coffea.dataset_tools.filespec.DataGroupSpec` as compact ``rich`` trees
(for the terminal) and as collapsible ``<details>`` HTML (for Jupyter). The spec
models auto-hook these via ``__rich__`` / ``_repr_html_`` so a bare spec renders
in both environments.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

from rich.tree import Tree

if TYPE_CHECKING:
    from rich.console import Console

    from coffea.dataset_tools.filespec import DataGroupSpec, DatasetSpec

_SPARK = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# Form leaf extraction (dtype + title aware)
# ---------------------------------------------------------------------------
class Leaf(NamedTuple):
    path: str
    dtype: str | None
    doc: str | None


_WRAPPERS = frozenset(
    {
        "ListOffsetArray",
        "ListArray",
        "RegularArray",
        "ByteMaskedArray",
        "BitMaskedArray",
        "UnmaskedArray",
        "IndexedOptionArray",
        "IndexedArray",
    }
)


def _form_to_dict(form: Any) -> dict | None:
    if form is None:
        return None
    if isinstance(form, dict):
        return form
    if isinstance(form, str):
        return json.loads(form)
    # awkward Form
    return form.to_dict()


def iter_leaves(node: dict, path: str = "", doc: str | None = None) -> list[Leaf]:
    """Return the leaf columns of an awkward form dict as (path, dtype, doc)."""
    cls = node.get("class", "")
    node_doc = (node.get("parameters") or {}).get("__doc__", doc)
    if cls == "RecordArray":
        fields = node.get("fields")
        contents = node.get("contents", [])
        out: list[Leaf] = []
        for i, content in enumerate(contents):
            field = fields[i] if fields else str(i)
            child = f"{path}.{field}" if path else field
            out.extend(iter_leaves(content, child, node_doc))
        return out
    if cls in _WRAPPERS:
        return iter_leaves(node.get("content", {}), path, node_doc)
    if cls == "UnionArray":
        out = []
        for c in node.get("contents", []):
            out.extend(iter_leaves(c, path, node_doc))
        return out
    # Leaf: NumpyArray, EmptyArray, ...
    if not path:
        return []
    return [Leaf(path, node.get("primitive"), node_doc)]


def column_leaves(form: Any) -> list[Leaf]:
    """Public: extract leaf columns (path, dtype, doc) from a spec form."""
    d = _form_to_dict(form)
    return iter_leaves(d) if d is not None else []


# ---------------------------------------------------------------------------
# Grouping (dot-nesting, then NanoAOD underscore-prefix convention)
# ---------------------------------------------------------------------------
def _group_leaves(
    leaves: list[Leaf], min_group: int = 3
) -> tuple[dict[str, list[tuple[str, Leaf]]], list[Leaf]]:
    """Group leaves into collections; return (groups{head: [(tail, leaf)]}, singletons)."""
    dot_groups: dict[str, list[tuple[str, Leaf]]] = defaultdict(list)
    flat: list[Leaf] = []
    for leaf in leaves:
        head, _, tail = leaf.path.partition(".")
        if tail:
            dot_groups[head].append((tail, leaf))
        else:
            flat.append(leaf)

    # underscore-prefix grouping for flat NanoAOD-style names
    buckets: dict[str, list[tuple[str, Leaf]]] = defaultdict(list)
    singletons: list[Leaf] = []
    for leaf in flat:
        head, sep, tail = leaf.path.partition("_")
        if sep and tail:
            buckets[head].append((tail, leaf))
        else:
            singletons.append(leaf)

    groups = dict(dot_groups)
    for head, members in buckets.items():
        if len(members) >= min_group:
            groups.setdefault(head, []).extend(members)
        else:
            singletons.extend(leaf for _, leaf in members)
    return groups, singletons


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _step_sizes(files: dict) -> list[int]:
    sizes: list[int] = []
    for spec in files.values():
        if spec.steps:
            sizes.extend(stop - start for start, stop in spec.steps)
    return sizes


def _sparkline(sizes: list[int], width: int = 16) -> str:
    if not sizes:
        return ""
    lo, hi = min(sizes), max(sizes)
    if lo == hi:
        return _SPARK[7] * min(width, len(sizes))
    bucket = (hi - lo) / width
    counts = [0] * width
    for s in sizes:
        i = min(int((s - lo) / bucket), width - 1)
        counts[i] += 1
    peak = max(counts)
    return "".join(_SPARK[round(c / peak * 7)] for c in counts)


def _is_concrete(fs: Any) -> bool:
    """A fully-preprocessed file: steps, num_entries and uuid all known."""
    return (
        getattr(fs, "steps", None) is not None
        and getattr(fs, "num_entries", None) is not None
        and getattr(fs, "uuid", None) is not None
    )


def _dataset_status(spec: DatasetSpec) -> tuple[bool, int, int]:
    """Return (all_preprocessed, n_concrete, n_files)."""
    files = spec.files
    n = len(files)
    n_concrete = sum(_is_concrete(v) for v in files.values())
    all_pre = n > 0 and n_concrete == n and spec.form is not None
    return all_pre, n_concrete, n


def _tail(path: str, keep: int = 60) -> str:
    s = str(path)
    return ("…" + s[-keep:]) if len(s) > keep else s


# ---------------------------------------------------------------------------
# rich rendering
# ---------------------------------------------------------------------------
def _add_columns(parent: Tree, form: Any, inline_limit: int = 6) -> None:
    leaves = column_leaves(form)
    if not leaves:
        parent.add("[dim]no form[/dim]")
        return
    parent.label = f"{parent.label}  [dim]({len(leaves)} total)[/dim]"
    groups, singletons = _group_leaves(leaves)
    for head in sorted(groups):
        members = groups[head]
        tails = [t for t, _ in members]
        if len(tails) <= inline_limit:
            parent.add(f"[cyan]{head}[/cyan][dim]{{{', '.join(tails)}}}[/dim]")
        else:
            node = parent.add(f"[cyan]{head}[/cyan]  [dim]{len(tails)} fields[/dim]")
            for tail, leaf in members[:inline_limit]:
                dt = f"[dim]: {leaf.dtype}[/dim]" if leaf.dtype else ""
                node.add(f"{tail}{dt}")
            if len(tails) > inline_limit:
                node.add(f"[dim]… {len(tails) - inline_limit} more[/dim]")
    if singletons:
        names = ", ".join(sorted(leaf.path for leaf in singletons))
        parent.add(f"[dim]other:[/dim]  [green]{names}[/green]")


def dataset_tree(
    spec: DatasetSpec, name: str = "dataset", show_columns: bool = True
) -> Tree:
    """Build a rich Tree for a single DatasetSpec."""
    files = spec.files
    keys = list(files.keys())
    n = len(keys)
    entries = spec.num_entries
    selected = spec.num_selected_entries
    all_pre, n_concrete, _ = _dataset_status(spec)

    status = (
        "[green]✓ preprocessed[/green]"
        if all_pre
        else f"[yellow]◔ {n_concrete}/{n} preprocessed[/yellow]"
    )
    label = (
        f"[bold]{name}[/bold]"
        f"  [dim]{spec.format or '?'} | {n} file{'s' if n != 1 else ''}"
        + (f" | {entries:,} entries" if entries is not None else "")
        + "[/dim]  "
        + status
    )
    tree = Tree(label)

    fb = tree.add("[yellow]files[/yellow]")
    if keys:
        fb.add(_tail(keys[0]))
        if n > 2:
            fb.add(f"[dim]  ⋮  ({n - 2} more)[/dim]")
        if n > 1:
            fb.add(_tail(keys[-1]))

    sizes = _step_sizes(files)
    if sizes:
        lo, hi = min(sizes), max(sizes)
        med = statistics.median(sizes)
        sel = ""
        if selected is not None and entries:
            sel = f"  [dim]selected {selected:,}/{entries:,} ({selected / entries:.0%})[/dim]"
        sb = tree.add("[yellow]steps[/yellow]")
        sb.add(
            f"{_sparkline(sizes)}  min={lo:,} med={med:,.0f} max={hi:,} n={len(sizes):,}{sel}"
        )

    if spec.metadata:
        mb = tree.add("[yellow]metadata[/yellow]")
        for k, v in spec.metadata.items():
            mb.add(f"[dim]{k}:[/dim] {v!r}")

    if show_columns:
        cb = tree.add("[yellow]columns[/yellow]")
        _add_columns(cb, spec.form)

    return tree


def datagroup_tree(spec: DataGroupSpec, show_columns: bool = True) -> Tree:
    """Build a rich Tree for a DataGroupSpec."""
    datasets = spec.root
    total = spec.num_entries
    n = len(datasets)
    label = (
        f"[bold]DataGroupSpec[/bold]"
        f"  [dim]{n} dataset{'s' if n != 1 else ''}"
        + (f" | {total:,} total entries" if total is not None else "")
        + "[/dim]"
    )
    root = Tree(label)
    for ds_name, ds in datasets.items():
        root.add(dataset_tree(ds, name=ds_name, show_columns=show_columns))
    return root


def print_dataset(
    spec: DatasetSpec,
    name: str = "dataset",
    show_columns: bool = True,
    console: Console | None = None,
) -> None:
    """Print a DatasetSpec as a compact rich tree."""
    from coffea.util import coffea_console

    (console or coffea_console).print(dataset_tree(spec, name, show_columns))


def print_datagroup(
    spec: DataGroupSpec,
    show_columns: bool = True,
    console: Console | None = None,
) -> None:
    """Print a DataGroupSpec as a compact rich tree."""
    from coffea.util import coffea_console

    (console or coffea_console).print(datagroup_tree(spec, show_columns))


# ---------------------------------------------------------------------------
# HTML rendering (collapsible <details>, native, no JS)
# ---------------------------------------------------------------------------
_HTML_STYLE = (
    "font-family:var(--jp-code-font-family,monospace);font-size:0.85em;line-height:1.4;"
)


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _columns_html(form: Any) -> str:
    leaves = column_leaves(form)
    if not leaves:
        return "<div style='opacity:0.6'>no form</div>"
    groups, singletons = _group_leaves(leaves)
    parts = [
        f"<summary>columns <span style='opacity:0.6'>({len(leaves)})</span></summary>"
    ]
    parts.append("<div style='margin-left:1em'>")
    for head in sorted(groups):
        members = groups[head]
        rows = "".join(
            f"<div><span style='color:#268bd2'>{_esc(tail)}</span>"
            + (
                f" <span style='opacity:0.6'>: {_esc(leaf.dtype)}</span>"
                if leaf.dtype
                else ""
            )
            + (
                f" <span style='opacity:0.5'>“{_esc(leaf.doc)}”</span>"
                if leaf.doc
                else ""
            )
            + "</div>"
            for tail, leaf in members
        )
        parts.append(
            f"<details><summary><b style='color:#2aa198'>{_esc(head)}</b>"
            f" <span style='opacity:0.6'>({len(members)})</span></summary>"
            f"<div style='margin-left:1em'>{rows}</div></details>"
        )
    if singletons:
        names = ", ".join(
            _esc(leaf.path) for leaf in sorted(singletons, key=lambda x: x.path)
        )
        parts.append(f"<div style='opacity:0.7'>other: {names}</div>")
    parts.append("</div>")
    return "".join(parts)


def _dataset_html_body(spec: DatasetSpec) -> str:
    files = spec.files
    keys = list(files.keys())
    n = len(keys)
    entries = spec.num_entries
    selected = spec.num_selected_entries
    parts: list[str] = []

    # files
    file_rows = "".join(f"<div>{_esc(_tail(k, 90))}</div>" for k in keys)
    parts.append(
        f"<details><summary>files <span style='opacity:0.6'>({n})</span></summary>"
        f"<div style='margin-left:1em'>{file_rows}</div></details>"
    )

    # steps
    sizes = _step_sizes(files)
    if sizes:
        lo, hi = min(sizes), max(sizes)
        med = statistics.median(sizes)
        sel = ""
        if selected is not None and entries:
            sel = f" · selected {selected:,}/{entries:,} ({selected / entries:.0%})"
        parts.append(
            "<details><summary>steps</summary>"
            f"<div style='margin-left:1em'><code>{_sparkline(sizes)}</code>"
            f" min={lo:,} med={med:,.0f} max={hi:,} n={len(sizes):,}{_esc(sel)}</div></details>"
        )

    # metadata
    if spec.metadata:
        rows = "".join(
            f"<div><span style='opacity:0.6'>{_esc(k)}:</span> {_esc(repr(v))}</div>"
            for k, v in spec.metadata.items()
        )
        parts.append(
            "<details><summary>metadata</summary>"
            f"<div style='margin-left:1em'>{rows}</div></details>"
        )

    # columns
    parts.append(f"<details>{_columns_html(spec.form)}</details>")
    return "".join(parts)


def dataset_html(spec: DatasetSpec, name: str = "dataset", open_: bool = True) -> str:
    """Render a DatasetSpec as collapsible HTML."""
    entries = spec.num_entries
    all_pre, n_concrete, n = _dataset_status(spec)
    badge = (
        "<span style='color:#859900'>✓ preprocessed</span>"
        if all_pre
        else f"<span style='color:#b58900'>◔ {n_concrete}/{n} preprocessed</span>"
    )
    summary = (
        f"<b>{_esc(name)}</b> <span style='opacity:0.7'>{_esc(spec.format or '?')} · "
        f"{n} file{'s' if n != 1 else ''}"
        + (f" · {entries:,} entries" if entries is not None else "")
        + f"</span> {badge}"
    )
    body = _dataset_html_body(spec)
    return (
        f"<div style='{_HTML_STYLE}'><details{' open' if open_ else ''}>"
        f"<summary>{summary}</summary>"
        f"<div style='margin-left:1em'>{body}</div></details></div>"
    )


def datagroup_html(spec: DataGroupSpec) -> str:
    """Render a DataGroupSpec as collapsible HTML."""
    datasets = spec.root
    n = len(datasets)
    total = spec.num_entries
    summary = (
        f"<b>DataGroupSpec</b> <span style='opacity:0.7'>{n} dataset{'s' if n != 1 else ''}"
        + (f" · {total:,} total entries" if total is not None else "")
        + "</span>"
    )
    bodies = "".join(
        dataset_html(ds, name=ds_name, open_=(n == 1))
        for ds_name, ds in datasets.items()
    )
    return (
        f"<div style='{_HTML_STYLE}'><details open><summary>{summary}</summary>"
        f"<div style='margin-left:1em'>{bodies}</div></details></div>"
    )
