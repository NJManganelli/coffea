# Plan: Arrow Flight ingestion for NanoEventsFactory (`from_flight`)

Branch: `parquet_dask_and_more` (coffea). Companion consumer for the `column_join`
Arrow Flight delivery service, but designed as a **generic** Flight ingestion path
so it is upstreamable to scikit-hep/coffea.

## Motivation

`column_join` already serves joined NanoAOD columns over Arrow Flight
(`ColumnJoinFlightServer`: one `FlightEndpoint` per output file, capitalization-restored
Arrow schema, column projection via JSON descriptor/ticket commands). Today the only
ways to get that output into coffea are workarounds:

- materialize to parquet + `to_datagroupspec()` + `apply_to_fileset` (disk round trip), or
- `read_flight` → `ak.from_arrow` → `ak.to_parquet` → `NanoEventsFactory.from_parquet`
  (notebook Example 3 — a literal disk round trip of data that already arrived in memory).

`NanoEventsFactory.from_flight(...)` removes the round trip and gives Flight the same
three modes as parquet: **eager**, **virtual** (touched columns only, fetched over the
wire with server-side projection), and **dask** (one partition per endpoint, typetracer
column projection driving the projected `DoGet` — the distributed-delivery topology
Flight was designed for).

## Design

### Entry point (coffea `factory.py`)

```python
NanoEventsFactory.from_flight(
    location,                    # "grpc://host:port" or an existing flight.FlightClient
    descriptor,                  # flight.FlightDescriptor | bytes | str | dict
                                 #   dict/str convenience → FlightDescriptor.for_command(json)
    *,
    mode="virtual",              # eager | virtual | dask
    schemaclass=NanoAODSchema,
    metadata=None,
    entry_start=None, entry_stop=None,     # eager/virtual only
    buffer_cache=None, access_log=None,    # parity with from_parquet
    descriptor_for_columns=None, # Callable[[list[str] | None], FlightDescriptor]
                                 # projection hook — protocol-specific; default provided
                                 # for the JSON-command convention {"dataset":…,"columns":…}
    client_kwargs=None,          # tls_root_certs / cert_chain / private_key → FlightClient
    call_options=None,           # flight.FlightCallOptions (bearer headers etc.)
)
```

Column projection is **not** part of the Flight standard, so it is a pluggable hook.
When `descriptor` is given as a dict/str in the `{"dataset": ..., "columns": ...}`
convention (used by column_join), a default `descriptor_for_columns` is synthesized
that rebuilds the command with the projected column list. A raw
`FlightDescriptor`/bytes descriptor with no hook simply disables wire-level projection
(virtual/dask still work; they fetch the advertised schema's full column set per read).

### Base form / identity

`GetFlightInfo` returns the (already projected, capitalization-restored) `pa.Schema`.
- Base awkward form: prefer serialized form JSON in schema metadata key `b"form"`
  (same key the parquet path already honors, `mapping/parquet.py::_extract_base_form`);
  fall back to `ak.from_arrow_schema(info.schema)` (verified available).
- `uuid`: schema metadata `b"uuid"` if present, else a deterministic
  `uuid5(NAMESPACE_URL, location + descriptor bytes)`.
- `object_path`: schema metadata `b"object_path"` else the descriptor repr.
- `num_rows`: `FlightInfo.total_records` when ≥ 0; eager mode can fall back to the
  materialized table length; virtual mode requires it (documented error otherwise).

The `_lazify_form` / `PreloadedSourceMapping._extract_base_form` route (what the
parquet dask path reuses) is the preferred implementation seam — Flight delivers
"a dict of in-memory columns", exactly what the preloaded machinery models.

### Eager / virtual modes

New `src/coffea/nanoevents/mapping/flight.py`, mirroring `mapping/parquet.py`:

- `TrivialFlightOpener(UUIDOpener)` — holds client/descriptor/call-options; `open_uuid`
  returns a shim exposing per-column reads.
- `FlightSourceMapping(BaseSourceMapping)` — `get_column_handle`/`extract_column` over
  Arrow columns (convert via `ak.from_arrow`), `_extract_base_form` from the FlightInfo
  schema as above.
- **Eager**: one `DoGet` per endpoint, `pa.concat_tables`, columns preloaded.
- **Virtual**: `BaseSourceMapping.__getitem__` already returns `partial(_getitem, key)`
  callables when `virtual=True`; the column handle triggers, per column, a **projected**
  `DoGet` (single-column ticket via the projection hook) with a per-mapping cache of
  fetched columns. Without a projection hook, first touch fetches the whole table once
  and caches it (lazy conversion, no lazy wire transfer — documented).
- `entry_start/entry_stop` implemented by slicing after fetch (Flight has no row-range
  pushdown in this protocol).

### Dask mode

All in coffea (no dask-awkward changes — precedent: uproot implements its own io_func
and calls `dask_awkward.from_map`, which is public):

- `_map_schema_flight(_map_schema_base)` in `factory.py`: `__call__(form) → (schema_form, self)`
  and `load_buffers(columns, keys, start, stop, options)` — near-copy of
  `_map_schema_parquet` (the preloaded/`_TranslatedMapping` pattern), with partition key
  derived from `options` = `{"location": ..., "ticket": ...}`.
- `FromFlightFn` io function (in `mapping/flight.py` or `nanoevents/flight.py`),
  templated on dask-awkward's `FromParquetFormMappedFn` (`a39ebba`), implementing
  `__call__`, `mock`, `mock_empty`, `prepare_for_projection`, `necessary_columns`,
  `project`, `project_keys`, `project_manually` (protocols in
  `dask_awkward/layers/layers.py`). One partition per `FlightEndpoint`; `GetFlightInfo`
  runs once at graph-construction time to enumerate endpoints and grab the schema/form.
- `project(...)` translates touched buffer keys → raw column names via
  `keys_for_buffer_keys`, then rebuilds per-partition tickets through the projection
  hook (or re-issues `GetFlightInfo` with a projected descriptor). Target:
  `dak.report_necessary_columns(events.Jet.pt) == {"nJet", "Jet_pt"}` and only those
  columns crossing the wire on compute.
- Endpoints with populated `locations` are honored (fresh client per endpoint —
  multi-serve-node topology); empty locations reuse the factory's client. Note: each
  dask worker constructs its own client from the endpoint URI (clients are not
  serialized into the graph — store URI + client_kwargs, connect lazily in `__call__`).

### Tests (coffea `tests/test_nanoevents_flight.py`)

Self-contained in-process test server (`flight.FlightServerBase`, `grpc://127.0.0.1:0`,
`.port`, `shutdown()` — same idiom as column_join's `tests/test_flight.py`), serving
the existing `tests/samples/nano_dy.parquet` / `.extensionarray.parquet` fixtures with
the JSON command/ticket convention, column projection, multi-endpoint (split the sample
into 2 files/endpoints), and optional form-in-schema-metadata:

1. `test_read_nanomc_flight[eager|virtual]` — mirror `test_read_nanomc` physics asserts.
2. Cross-mode equality: flight eager/virtual/dask values == `from_parquet` virtual values.
3. `test_flight_dask_projection` — `report_necessary_columns` returns raw column names;
   server-side access log records that only projected columns were requested in virtual
   and dask modes.
4. Multi-endpoint concat order, `entry_start/stop`, no-hook fallback, metadata-form
   preference, `total_records=-1` error path for virtual.

## column_join follow-through (separate repo, after coffea lands)

1. Server enrichment: attach `form` (awkward form JSON, original capitalization,
   projected consistently with the advertised schema), `uuid`, `object_path` to the
   advertised `pa.Schema` metadata (both `get_flight_info` and `get_schema`, all
   backends); populate per-endpoint `total_records` where known.
2. Export a `descriptor_for(dataset, columns)` helper matching coffea's hook signature.
3. Notebook `ColumnJoinExamples.ipynb`: replace Example 3's
   `ak.to_parquet`/`from_parquet` round trip with `NanoEventsFactory.from_flight`;
   extend the Example 4 capstone (or add Example 5) with dask-mode Flight consumption +
   `report_necessary_columns` demonstrating wire-level projection.
4. Tests: schema-metadata assertions in `tests/test_flight.py`; an integration test
   consuming a live in-process `ColumnJoinFlightServer` through
   `NanoEventsFactory.from_flight` in all three modes.

## Deferred (explicitly out of scope for this pass)

- `DatasetSpec`/`apply_to_fileset` `format="flight"` routing (needs filespec URI design).
- Row-range pushdown over Flight; streaming/batch-wise partitioning within an endpoint.
- Auth beyond passthrough kwargs (grid X.509/token management stays in column_join).
- dask-awkward upstream changes (none required).

## Environment

Editable installs in the column-join pixi env:
`cd /Users/nmangane/servicex_claude/column-join && pixi run python -m pytest <coffea>/tests/test_nanoevents_flight.py`
