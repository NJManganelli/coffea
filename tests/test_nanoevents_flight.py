import json

import awkward as ak
import pytest

pytest.importorskip("pyarrow.flight")

import pyarrow as pa  # noqa: E402
import pyarrow.flight as flight  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory  # noqa: E402


def genroundtrips(genpart):
    assert ak.all(genpart.children.parent.pdgId == genpart.pdgId)
    assert ak.all(
        ak.any(
            genpart.parent.children.pdgId == genpart.pdgId, axis=-1, mask_identity=True
        )
    )
    assert ak.all(genpart.distinctParent.pdgId != genpart.pdgId)
    assert ak.all(genpart.distinctChildren.pdgId != genpart.pdgId)


def crossref(events):
    assert ak.all(events.Jet.matched_muons.matched_jet.pt == events.Jet.pt)
    assert ak.all(
        events.Electron.matched_photon.matched_electron.r9 == events.Electron.r9
    )


class _FlightTestServer(flight.FlightServerBase):
    """In-process Flight server over one or more parquet tables.

    Honors the ``{"dataset": ..., "columns": ...}`` JSON command/ticket
    convention with column projection, splits the served rows across one
    endpoint per table, records requested columns, and optionally advertises
    a serialized awkward form and a negative ``total_records``.
    """

    def __init__(self, location, tables, *, form=None, total_records=None):
        super().__init__(location)
        self._tables = tables
        self._form = form
        self._total_records = total_records
        self.requests = []

    def _schema(self, columns):
        base = self._tables[0]
        schema = base.schema if columns is None else base.select(columns).schema
        meta = dict(schema.metadata or {})
        if self._form is not None:
            meta[b"form"] = self._form.to_json().encode()
            schema = schema.with_metadata(meta)
        elif meta:
            schema = schema.with_metadata(meta)
        return schema

    def _project(self, table, columns):
        return table if columns is None else table.select(columns)

    def _columns(self, payload_bytes):
        payload = json.loads(payload_bytes)
        return payload.get("columns", None)

    def get_flight_info(self, context, descriptor):
        columns = self._columns(descriptor.command)
        schema = self._schema(columns)
        n_rows = sum(t.num_rows for t in self._tables)
        endpoints = []
        for i in range(len(self._tables)):
            ticket = flight.Ticket(
                json.dumps({"columns": columns, "part": i}).encode()
            )
            endpoints.append(flight.FlightEndpoint(ticket, []))
        total = self._total_records if self._total_records is not None else n_rows
        return flight.FlightInfo(schema, descriptor, endpoints, total, -1)

    def do_get(self, context, ticket):
        payload = json.loads(ticket.ticket)
        columns = payload.get("columns", None)
        part = payload.get("part", None)
        self.requests.append(columns)
        if part is None:
            table = pa.concat_tables(self._tables)
        else:
            table = self._tables[part]
        return flight.RecordBatchStream(self._project(table, columns))


def _serve(tables, *, form=None, total_records=None):
    server = _FlightTestServer(
        "grpc://127.0.0.1:0", tables, form=form, total_records=total_records
    )
    location = f"grpc://127.0.0.1:{server.port}"
    return server, location


@pytest.fixture
def single_table(tests_directory):
    return pq.read_table(f"{tests_directory}/samples/nano_dy.parquet")


@pytest.fixture
def extension_table(tests_directory):
    return pq.read_table(f"{tests_directory}/samples/nano_dy.extensionarray.parquet")


@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_read_nanomc_flight(single_table, mode):
    server, location = _serve([single_table])
    try:
        factory = NanoEventsFactory.from_flight(
            location,
            {"dataset": "nano"},
            schemaclass=NanoAODSchema,
            mode=mode,
        )
        events = factory.events()

        genroundtrips(events.GenPart)
        genroundtrips(events.GenPart[events.GenPart.eta > 0])

        assert ak.all(
            (abs(events.Electron.matched_gen.pdgId) == 11)
            | (events.Electron.matched_gen.pdgId == 22)
        )
        assert ak.all(abs(events.Muon.matched_gen.pdgId) == 13)

        crossref(events[ak.num(events.Jet) > 2])
        crossref(events)

        assert ak.to_list(events[[]].Photon.mass) == []
        assert ak.any(events.Photon.isTight, axis=1).tolist()[:9] == [
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
    finally:
        server.shutdown()


@pytest.mark.parametrize("mode", ["eager", "virtual", "dask"])
def test_flight_matches_parquet(tests_directory, single_table, mode):
    path = f"{tests_directory}/samples/nano_dy.parquet"
    ref = NanoEventsFactory.from_parquet(
        path, schemaclass=NanoAODSchema, mode="virtual"
    ).events()

    server, location = _serve([single_table])
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode=mode
        )
        events = factory.events()
        jet_pt = events.Jet.pt.compute() if mode == "dask" else events.Jet.pt
        met = events.MET.pt.compute() if mode == "dask" else events.MET.pt

        assert ak.all(ak.flatten(jet_pt) == ak.flatten(ref.Jet.pt))
        assert ak.all(met == ref.MET.pt)
    finally:
        server.shutdown()


def test_flight_dask_projection(single_table):
    dak = pytest.importorskip("dask_awkward")

    server, location = _serve([single_table])
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode="dask"
        )
        events = factory.events()

        (necessary,) = dak.report_necessary_columns(events.Jet.pt).values()
        assert necessary == frozenset({"nJet", "Jet_pt"})

        server.requests.clear()
        events.Jet.pt.compute()
        assert len(server.requests) == 1
        assert set(server.requests[0]) == {"nJet", "Jet_pt"}
    finally:
        server.shutdown()


def test_flight_virtual_projection(single_table):
    server, location = _serve([single_table])
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode="virtual"
        )
        events = factory.events()
        server.requests.clear()
        ak.sum(ak.flatten(events.Jet.pt))
        requested = {c for req in server.requests for c in (req or [])}
        assert "Jet_pt" in requested
        assert "Muon_pt" not in requested
    finally:
        server.shutdown()


def test_flight_multi_endpoint(single_table):
    dak = pytest.importorskip("dask_awkward")

    n = single_table.num_rows
    half = n // 2
    tables = [single_table.slice(0, half), single_table.slice(half)]

    ref = pa.concat_tables(tables)
    server, location = _serve(tables)
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode="dask"
        )
        events = factory.events()
        assert events.npartitions == 2

        event_ids = ak.to_list(events.event.compute())
        assert event_ids == ref.column("event").to_pylist()
    finally:
        server.shutdown()


def test_flight_entry_range(single_table):
    # BaseSchema avoids cross-reference transforms whose global indices do not
    # survive naive row-slicing (the same limitation applies to from_parquet).
    server, location = _serve([single_table])
    try:
        factory = NanoEventsFactory.from_flight(
            location,
            {"dataset": "nano"},
            schemaclass=BaseSchema,
            mode="eager",
            entry_start=5,
            entry_stop=15,
        )
        events = factory.events()
        assert len(events) == 10
        expected = single_table.column("event").to_pylist()[5:15]
        assert ak.to_list(events.event) == expected
    finally:
        server.shutdown()


def test_flight_no_hook_fallback(single_table):
    # A command that is not the {"dataset": ...} convention synthesizes no hook,
    # so wire-level projection is disabled and the whole table is fetched once.
    descriptor = flight.FlightDescriptor.for_command(json.dumps({}).encode())
    server, location = _serve([single_table])
    try:
        factory = NanoEventsFactory.from_flight(
            location,
            descriptor,
            schemaclass=NanoAODSchema,
            mode="virtual",
        )
        assert factory._mapping._file_handle._descriptor_for_columns is None
        events = factory.events()
        server.requests.clear()
        ak.sum(ak.flatten(events.Jet.pt))
        # Without projection, the whole table is fetched once and cached.
        assert server.requests == [None]
        ak.sum(ak.flatten(events.Muon.pt))
        assert server.requests == [None]
    finally:
        server.shutdown()


def test_flight_form_metadata_preference(single_table):
    form = ak.from_arrow_schema(single_table.schema)
    server, location = _serve([single_table], form=form)
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode="eager"
        )
        events = factory.events()
        assert ak.sum(ak.num(events.Jet)) == ak.sum(
            ak.num(ak.from_arrow(single_table.column("Jet_pt")))
        )
    finally:
        server.shutdown()


def test_flight_dask_nullable_extension_columns(tmp_path):
    # Regression: flight_dask built its base form via awkward.from_arrow_schema
    # applied directly to the advertised schema. A column_join-style producer
    # serves plain (Hive/Trino-written) Arrow columns -- pa.field(name, type)
    # defaults nullable=True regardless of whether any value is actually null
    # -- retyped to awkward's Arrow-extension (AwkwardArrowType) on the
    # advertised schema. For that Arrow-nullable=True + extension-type
    # combination, awkward.from_arrow_schema reconstructs list columns as
    # BitMaskedArray-wrapped forms with no form_key -- not the ListOffsetArray
    # (+ "!load"/"!load,!content" form_key) shape NanoEvents' lazy buffer
    # loading requires. extract_flight_base_form (what eager/virtual already
    # use, and what the advertised schema's b"form" metadata is for) produces
    # the correct shape; dask mode must use it too. Without the fix this
    # raises "There are missing event ID fields" while NanoAODSchema tries to
    # build collections from the malformed form.
    pytest.importorskip("dask_awkward")

    arr = ak.Array(
        {
            "run": [1, 1, 1],
            "luminosityBlock": [7, 7, 7],
            "event": [10, 11, 12],
            "nJet": [2, 0, 1],
            "Jet_pt": [[30.0, 25.0], [], [40.0]],
            "Jet_eta": [[0.1, -0.2], [], [1.1]],
            "Jet_phi": [[0.5, -0.5], [], [1.5]],
            "Jet_mass": [[5.0, 4.0], [], [6.0]],
        }
    )
    # Retype to nullable=True extension columns without touching buffers --
    # the same operation column_join's restore_schema_capitalization performs
    # when relabeling a plain (Hive-written) nullable Arrow schema onto an
    # awkward-extension-typed advertised schema.
    extension_table = ak.to_arrow_table(arr)
    nullable_fields = [
        pa.field(f.name, f.type, nullable=True) for f in extension_table.schema
    ]
    table = pa.Table.from_arrays(
        extension_table.columns, schema=pa.schema(nullable_fields)
    )

    server, location = _serve([table], form=arr.layout.form)
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode="dask"
        )
        events = factory.events()
        assert ak.to_list(events.Jet.pt.compute()) == [[30.0, 25.0], [], [40.0]]
    finally:
        server.shutdown()


def test_flight_virtual_requires_num_rows(single_table):
    server, location = _serve([single_table], total_records=-1)
    try:
        with pytest.raises(ValueError, match="virtual mode requires a known row count"):
            NanoEventsFactory.from_flight(
                location,
                {"dataset": "nano"},
                schemaclass=NanoAODSchema,
                mode="virtual",
            )
    finally:
        server.shutdown()


def test_flight_eager_num_rows_fallback(single_table):
    server, location = _serve([single_table], total_records=-1)
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode="eager"
        )
        events = factory.events()
        assert len(events) == single_table.num_rows
    finally:
        server.shutdown()


def test_flight_extensionarray(extension_table):
    server, location = _serve([extension_table])
    try:
        factory = NanoEventsFactory.from_flight(
            location, {"dataset": "nano"}, schemaclass=NanoAODSchema, mode="eager"
        )
        events = factory.events()
        assert len(events) == extension_table.num_rows
        crossref(events)
    finally:
        server.shutdown()
