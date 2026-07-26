"""Textual application for interactively exploring dataset specs.

This module imports ``textual`` at import time and is therefore only imported
lazily, by :func:`coffea.dataset_tools._explore.explore` (which raises a helpful
message when ``textual`` -- the ``coffea[tui]`` extra -- is missing). All of the
non-UI logic lives in :mod:`coffea.dataset_tools._explore` and is unit-tested
separately; this file is a thin presentation layer over it.
"""

from __future__ import annotations

import json
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from coffea.dataset_tools._display import dataset_tree
from coffea.dataset_tools._explore import (
    _as_group,
    diff_specs,
    filter_columns,
    is_empty_diff,
    provenance,
    whatif,
)


def _fmt_summary(summary: dict[str, Any]) -> str:
    order = ["datasets", "files", "num_entries", "num_selected_entries", "partitions"]
    return "  ".join(
        f"{k}={summary[k]:,}" if isinstance(summary[k], int) else f"{k}={summary[k]}"
        for k in order
    )


class DatasetExplorerApp(App):
    """Explore a DatasetSpec / DataGroupSpec: overview, columns, what-if, diff."""

    CSS = """
    Input { margin: 1 0; }
    DataTable { height: 1fr; }
    #summary { padding: 1; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("e", "export", "Export what-if JSON"),
    ]

    def __init__(self, spec: Any, other: Any = None):
        super().__init__()
        self._group = _as_group(spec)
        self._names = list(self._group)
        self._current = self._names[0] if self._names else ""
        self._other = other
        self._whatif_result = self.dataset  # updated by the what-if tab

    @property
    def dataset(self):
        return self._group[self._current]

    # -- layout ------------------------------------------------------------- #
    def compose(self) -> ComposeResult:
        yield Header()
        if len(self._names) > 1:
            yield Select(
                [(n, n) for n in self._names], value=self._current, id="dataset-select"
            )
        with TabbedContent():
            with TabPane("Overview", id="tab-overview"):
                yield VerticalScroll(Static(id="overview"))
            with TabPane("Columns", id="tab-columns"):
                yield Input(placeholder="regex filter columns…", id="column-filter")
                yield DataTable(id="column-table")
            with TabPane("What-if", id="tab-whatif"):
                with Horizontal():
                    yield Input(placeholder="filter_files regex", id="wi-filter")
                    yield Input(placeholder="max_files", id="wi-maxfiles")
                    yield Input(placeholder="max_steps", id="wi-maxsteps")
                yield Static(id="summary")
            with TabPane("Provenance", id="tab-prov"):
                yield Input(placeholder="regex filter metadata keys…", id="prov-filter")
                yield DataTable(id="prov-table")
            with TabPane("Diff", id="tab-diff"):
                yield VerticalScroll(Static(id="diff"))
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#column-table", DataTable).add_columns(
            "column", "dtype", "title"
        )
        self.query_one("#prov-table", DataTable).add_columns("key", "value")
        self._refresh_all()

    # -- refresh helpers ---------------------------------------------------- #
    def _refresh_all(self) -> None:
        self._refresh_overview()
        self._refresh_columns("")
        self._refresh_whatif()
        self._refresh_provenance("")
        self._refresh_diff()

    def _refresh_overview(self) -> None:
        self.query_one("#overview", Static).update(
            dataset_tree(self.dataset, name=self._current)
        )

    def _refresh_columns(self, pattern: str) -> None:
        table = self.query_one("#column-table", DataTable)
        table.clear()
        for leaf in filter_columns(self.dataset, pattern):
            table.add_row(leaf.path, leaf.dtype or "", leaf.doc or "")

    def _refresh_whatif(self) -> None:
        def _int(widget_id: str) -> int | None:
            raw = self.query_one(widget_id, Input).value.strip()
            return int(raw) if raw.isdigit() else None

        filt = self.query_one("#wi-filter", Input).value.strip() or None
        result, summary = whatif(
            self.dataset,
            filter_name=filt,
            max_files=_int("#wi-maxfiles"),
            max_steps=_int("#wi-maxsteps"),
        )
        self._whatif_result = result
        self.query_one("#summary", Static).update(_fmt_summary(summary))

    def _refresh_provenance(self, pattern: str) -> None:
        table = self.query_one("#prov-table", DataTable)
        table.clear()
        for key, value in sorted(provenance(self.dataset, pattern or None).items()):
            table.add_row(str(key), repr(value))

    def _refresh_diff(self) -> None:
        widget = self.query_one("#diff", Static)
        if self._other is None:
            widget.update("Pass a second fileset (spec.explore(other)) to enable diff.")
            return
        diff = diff_specs(self._group, self._other)
        widget.update(
            "no differences"
            if is_empty_diff(diff)
            else json.dumps(diff, indent=2, default=str)
        )

    # -- events ------------------------------------------------------------- #
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value is not None:
            self._current = str(event.value)
            self._refresh_all()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "column-filter":
            self._refresh_columns(event.value)
        elif event.input.id == "prov-filter":
            self._refresh_provenance(event.value)
        elif event.input.id and event.input.id.startswith("wi-"):
            self._refresh_whatif()

    # -- actions ------------------------------------------------------------ #
    def action_export(self) -> None:
        path = f"whatif_{self._current or 'fileset'}.json"
        with open(path, "w") as handle:
            handle.write(self._whatif_result.model_dump_json(indent=2))
        self.notify(f"exported what-if fileset to {path}")

    def action_quit(self) -> None:
        self.exit()
