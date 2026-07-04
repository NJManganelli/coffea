import json
import uuid as uuidlib

import awkward
import numpy

from coffea.nanoevents.mapping.base import BaseSourceMapping, UUIDOpener
from coffea.nanoevents.mapping.parquet import arrow_schema_to_awkward_form
from coffea.nanoevents.util import quote, tuple_to_key

_FLIGHT_NAMESPACE = uuidlib.NAMESPACE_URL


def _location_uri(location):
    """Return the ``grpc://...`` URI string for a Flight location.

    Parameters
    ----------
        location : str or pyarrow.flight.FlightClient or pyarrow.flight.Location
            The Flight endpoint to describe.

    Returns
    -------
        str
            The URI used to (re)connect a client.
    """
    import pyarrow.flight as flight

    if isinstance(location, flight.Location):
        return location.uri.decode() if isinstance(location.uri, bytes) else location.uri
    if isinstance(location, flight.FlightClient):
        return None
    return str(location)


def normalize_descriptor(descriptor):
    """Coerce a user-supplied descriptor into a ``FlightDescriptor``.

    Parameters
    ----------
        descriptor : pyarrow.flight.FlightDescriptor or bytes or str or dict
            ``dict`` and ``str`` inputs are treated as a JSON command and wrapped
            with ``FlightDescriptor.for_command``; ``bytes`` are used verbatim as
            a command; a ``FlightDescriptor`` is returned unchanged.

    Returns
    -------
        pyarrow.flight.FlightDescriptor
            The normalized descriptor.
    """
    import pyarrow.flight as flight

    if isinstance(descriptor, flight.FlightDescriptor):
        return descriptor
    if isinstance(descriptor, dict):
        return flight.FlightDescriptor.for_command(json.dumps(descriptor).encode())
    if isinstance(descriptor, str):
        return flight.FlightDescriptor.for_command(descriptor.encode())
    if isinstance(descriptor, bytes):
        return flight.FlightDescriptor.for_command(descriptor)
    raise TypeError(f"Invalid descriptor type ({type(descriptor)})")


def _descriptor_command(descriptor):
    """Return the command payload of a descriptor, if it carries one."""
    command = getattr(descriptor, "command", None)
    return command


def make_descriptor_for_columns(descriptor):
    """Build a default ``descriptor_for_columns`` hook for the JSON convention.

    The column_join delivery service accepts a JSON command of the form
    ``{"dataset": ..., "columns": ...}``. When ``descriptor`` carries such a
    command, the returned hook rebuilds it with a projected column list; a
    column list of ``None`` requests the full advertised schema.

    Parameters
    ----------
        descriptor : pyarrow.flight.FlightDescriptor
            The base descriptor whose command is rewritten per projection.

    Returns
    -------
        Callable[[list[str] or None], pyarrow.flight.FlightDescriptor] or None
            The projection hook, or ``None`` if ``descriptor`` does not carry a
            ``{"dataset": ...}`` JSON command.
    """
    command = _descriptor_command(descriptor)
    if command is None:
        return None
    try:
        payload = json.loads(command)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or "dataset" not in payload:
        return None

    def descriptor_for_columns(columns):
        new_payload = dict(payload)
        if columns is None:
            new_payload.pop("columns", None)
        else:
            new_payload["columns"] = list(columns)
        return normalize_descriptor(new_payload)

    return descriptor_for_columns


def extract_flight_base_form(schema):
    """Extract a NanoEvents base form from a Flight ``pa.Schema``.

    Prefers a serialized awkward form JSON stored under the schema metadata key
    ``b"form"`` (the same convention the parquet path honors); otherwise falls
    back to per-field conversion via :func:`arrow_schema_to_awkward_form`.

    Parameters
    ----------
        schema : pyarrow.Schema
            The advertised (capitalization-restored, possibly projected) schema.

    Returns
    -------
        dict
            A ``RecordArray`` form dictionary suitable for a NanoEvents schema.
    """
    import warnings

    meta = schema.metadata or {}
    if b"form" in meta:
        form = json.loads(meta[b"form"])
        return _prepare_form_keys(form)

    column_forms = {}
    for field in schema:
        key = field.name
        fmeta = {} if field.metadata is None else field.metadata

        if "," in key or "!" in key:
            warnings.warn(
                f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
            )
            continue

        form = None
        if b"form" in fmeta:
            form = json.loads(fmeta[b"form"])
        else:
            form = json.loads(arrow_schema_to_awkward_form(field.type).to_json())

        if (
            form["class"].startswith("ListOffset")
            and form["content"]["class"] == "NumpyArray"  # noqa
        ):
            form["form_key"] = quote(f"{key},!load")
            form["content"]["form_key"] = quote(f"{key},!load,!content")
            if b"title" in fmeta:
                form["content"]["parameters"] = {"__doc__": fmeta[b"title"].decode()}
            elif "__doc__" not in form["content"].get("parameters", {}):
                form["content"]["parameters"] = {"__doc__": key}
        elif form["class"] == "NumpyArray":
            form["form_key"] = quote(f"{key},!load")
            if b"title" in fmeta:
                form["parameters"] = {"__doc__": fmeta[b"title"].decode()}
            elif "__doc__" not in form.get("parameters", {}):
                form["parameters"] = {"__doc__": key}
        else:
            warnings.warn(f"Skipping {key} as it is not interpretable by NanoEvents")
            continue
        column_forms[key] = form

    return {
        "class": "RecordArray",
        "contents": [item for item in column_forms.values()],
        "fields": [key for key in column_forms.keys()],
        "parameters": {"__doc__": "flightsource"},
        "form_key": "",
    }


def _prepare_form_keys(form):
    """Stamp ``!load`` form keys onto a serialized (raw) awkward form."""
    fields = form.get("fields", [])
    contents = form.get("contents", [])
    for key, content in zip(fields, contents):
        if (
            content["class"].startswith("ListOffset")
            and content["content"]["class"] == "NumpyArray"
        ):
            content["form_key"] = quote(f"{key},!load")
            content["content"]["form_key"] = quote(f"{key},!load,!content")
            if "__doc__" not in content["content"].get("parameters", {}):
                content["content"]["parameters"] = {"__doc__": key}
        elif content["class"] == "NumpyArray":
            content["form_key"] = quote(f"{key},!load")
            if "__doc__" not in content.get("parameters", {}):
                content["parameters"] = {"__doc__": key}
    form["form_key"] = ""
    form.setdefault("parameters", {})
    form["parameters"].setdefault("__doc__", "flightsource")
    return form


def derive_flight_identity(location, descriptor, schema, num_rows_default=None):
    """Derive ``(uuid, object_path, num_rows)`` for a Flight source.

    Parameters
    ----------
        location : str
            The Flight endpoint URI.
        descriptor : pyarrow.flight.FlightDescriptor
            The (normalized) descriptor identifying the dataset.
        schema : pyarrow.Schema
            The advertised schema, whose metadata may carry ``b"uuid"`` and
            ``b"object_path"``.
        num_rows_default : int or None, optional
            Fallback row count when the schema metadata omits it.

    Returns
    -------
        tuple[str, str, int or None]
            The derived uuid, object path, and row count.
    """
    meta = schema.metadata or {}
    descriptor_bytes = descriptor.serialize()

    fuuid = meta.get(b"uuid", None)
    if fuuid is None:
        seed = f"{location}".encode() + descriptor_bytes
        fuuid = str(uuidlib.uuid5(_FLIGHT_NAMESPACE, seed.hex()))
    else:
        fuuid = fuuid.decode("ascii")

    obj_path = meta.get(b"object_path", None)
    if obj_path is None:
        obj_path = repr(descriptor)
    else:
        obj_path = obj_path.decode("ascii")

    num_rows = meta.get(b"num_rows", None)
    if num_rows is not None:
        num_rows = int(num_rows)
    else:
        num_rows = num_rows_default

    return fuuid, obj_path, num_rows


class _FlightColumnSource:
    """A per-partition Flight column source with a projected-fetch cache.

    Fetched Arrow columns are cached by name. When a projection hook is
    available each first touch triggers a single-column projected ``DoGet``;
    otherwise the whole table is fetched once on first touch and cached.
    """

    def __init__(
        self,
        location,
        descriptor,
        schema,
        descriptor_for_columns=None,
        client=None,
        client_kwargs=None,
        call_options=None,
    ):
        self._location = location
        self._descriptor = descriptor
        self._schema = schema
        self._descriptor_for_columns = descriptor_for_columns
        self._client = client
        self._client_kwargs = client_kwargs or {}
        self._call_options = call_options
        self._columns = {}
        self._fetched_all = False

    def _connect(self):
        if self._client is not None:
            return self._client
        import pyarrow.flight as flight

        self._client = flight.FlightClient(self._location, **self._client_kwargs)
        return self._client

    def _do_get(self, descriptor):
        client = self._connect()
        info = client.get_flight_info(descriptor, self._call_options)
        tables = []
        for endpoint in info.endpoints:
            reader = client.do_get(endpoint.ticket, self._call_options)
            tables.append(reader.read_all())
        import pyarrow as pa

        return pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    def _fetch_all(self):
        if self._fetched_all:
            return
        descriptor = (
            self._descriptor_for_columns(None)
            if self._descriptor_for_columns is not None
            else self._descriptor
        )
        table = self._do_get(descriptor)
        for name in table.column_names:
            if name not in self._columns:
                self._columns[name] = table.column(name)
        self._fetched_all = True

    def __contains__(self, name):
        return name in self._schema.names

    def __getitem__(self, name):
        if name in self._columns:
            return self._columns[name]
        if self._descriptor_for_columns is not None:
            descriptor = self._descriptor_for_columns([name])
            table = self._do_get(descriptor)
            column = table.column(name)
            self._columns[name] = column
            return column
        self._fetch_all()
        return self._columns[name]


class TrivialFlightOpener(UUIDOpener):
    """Opener that yields per-partition Flight column sources.

    Parameters
    ----------
        uuid_pfnmap : dict
            Mapping from uuid to a preconstructed :class:`_FlightColumnSource`.
    """

    def __init__(self, uuid_pfnmap):
        super().__init__(uuid_pfnmap)

    def open_uuid(self, uuid):
        return self._uuid_pfnmap[uuid]


class FlightSourceMapping(BaseSourceMapping):
    """Source mapping backed by an Apache Flight endpoint.

    Columns are pulled over the wire (projected when a hook is available) and
    converted to awkward arrays via ``ak.from_arrow``. Eager mode preloads all
    columns; virtual mode fetches per-column on first touch.
    """

    _debug = False

    def __init__(
        self,
        fileopener,
        start,
        stop,
        cache=None,
        access_log=None,
        file_handle=None,
        virtual=False,
        buffer_cache=None,
    ):
        super().__init__(
            fileopener=fileopener,
            start=start,
            stop=stop,
            cache=cache,
            access_log=access_log,
            file_handle=file_handle,
            virtual=virtual,
            buffer_cache=buffer_cache,
        )

    @classmethod
    def _extract_base_form(cls, schema):
        return extract_flight_base_form(schema)

    def key_root(self):
        return "FlightSourceMapping:"

    def preload_column_source(self, uuid, path_in_source, source):
        """Register a preconstructed column source to save a re-open."""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    def get_column_handle(self, columnsource, name, allow_missing):
        if allow_missing:
            return columnsource[name] if name in columnsource else None
        return columnsource[name]

    def extract_column(self, columnhandle, start, stop, allow_missing, **kwargs):
        if allow_missing and columnhandle is None:
            return awkward.contents.IndexedOptionArray(
                awkward.index.Index64(numpy.full(stop - start, -1, dtype=numpy.int64)),
                awkward.contents.NumpyArray(numpy.array([], dtype=bool)),
            )
        elif not allow_missing and columnhandle is None:
            raise RuntimeError(
                "Received columnhandle of None when missing column in file is not allowed!"
            )

        the_array = awkward.from_arrow(columnhandle)[start:stop]

        if allow_missing:
            the_array = awkward.contents.IndexedOptionArray(
                awkward.index.Index64(numpy.arange(stop - start, dtype=numpy.int64)),
                awkward.contents.NumpyArray(the_array),
            )

        return the_array

    def __len__(self):
        return self._stop - self._start

    def __iter__(self):
        raise NotImplementedError


class FromFlightFn:
    """dask-awkward IO function reading one Flight endpoint per partition.

    Templated on dask-awkward's ``FromParquetFormMappedFn``: the mapped
    ("expected") form describes the user-facing NanoEvents array, and each of
    its buffers derives from one or more raw Flight columns declared by the
    ``form_mapping_info`` (a ``_map_schema_flight``). Column projection narrows
    the per-partition Flight ticket via the projection hook.

    Per-partition ``(location, ticket_bytes)`` arrive through the input
    iterable (as with ``FromParquetFn`` receiving a path); the io function
    holds only projection-relevant state (forms, common keys, hook), so a
    single instance drives every partition. Clients are never serialized into
    the graph: a fresh client is connected lazily inside ``__call__``.

    Parameters
    ----------
        base_form : awkward.forms.Form
            The raw (unmapped) Flight form.
        expected_form : awkward.forms.Form
            The schema-mapped NanoEvents form.
        form_mapping_info : ImplementsFormMappingInfo
            The ``_map_schema_flight`` object describing buffer loading.
        descriptor_bytes : bytes
            Serialized base descriptor (for provenance / partition keys).
        descriptor_for_columns : Callable or None
            Projection hook rebuilding a descriptor for a projected column list.
        common_keys : frozenset[str] or None
            The raw columns retained after projection (defaults to all).
        client_kwargs : dict or None
            Keyword arguments for constructing the per-partition client.
        call_options : pyarrow.flight.FlightCallOptions or None
            Call options passed to ``do_get``.
    """

    def __init__(
        self,
        *,
        base_form,
        expected_form,
        form_mapping_info,
        descriptor_bytes,
        descriptor_for_columns=None,
        common_keys=None,
        client_kwargs=None,
        call_options=None,
    ):
        self.base_form = base_form
        self.expected_form = expected_form
        self.form_mapping_info = form_mapping_info
        self.descriptor_bytes = descriptor_bytes
        self.descriptor_for_columns = descriptor_for_columns
        self.common_keys = frozenset(
            base_form.fields if common_keys is None else common_keys
        )
        self.client_kwargs = client_kwargs or {}
        self.call_options = call_options

    @property
    def return_report(self):
        return False

    @property
    def use_optimization(self):
        return True

    def _read_table(self, location, ticket_bytes, endpoint_index):
        import pyarrow.flight as flight

        client = flight.FlightClient(location, **self.client_kwargs)
        if self.descriptor_for_columns is not None:
            descriptor = self.descriptor_for_columns(sorted(self.common_keys))
            info = client.get_flight_info(descriptor, self.call_options)
            ticket = info.endpoints[endpoint_index].ticket
        else:
            ticket = flight.Ticket(ticket_bytes)
        reader = client.do_get(ticket, self.call_options)
        return reader.read_all()

    def __call__(self, partition):
        location, ticket_bytes, endpoint_index = partition
        table = self._read_table(location, ticket_bytes, endpoint_index)
        length = table.num_rows
        raw_columns = {}
        for name in table.column_names:
            if name in self.common_keys:
                raw_columns[name] = awkward.from_arrow(table.column(name))

        mapping = self.form_mapping_info.load_buffers(
            raw_columns,
            self.common_keys,
            0,
            length,
            {"location": location, "ticket": ticket_bytes},
        )

        from awkward._nplikes.numpy import Numpy

        nplike = Numpy.instance()

        container = {}
        for buffer_key, dtype in self.expected_form.expected_from_buffers(
            buffer_key=self.form_mapping_info.buffer_key
        ).items():
            keys_for_buffer = self.form_mapping_info.keys_for_buffer_keys(
                frozenset({buffer_key})
            )
            if all(k in self.common_keys for k in keys_for_buffer):
                container[buffer_key] = mapping[buffer_key]
            else:
                container[buffer_key] = awkward.typetracer.PlaceholderArray(
                    nplike=nplike,
                    shape=(awkward.typetracer.unknown_length,),
                    dtype=dtype,
                )

        return awkward.from_buffers(
            self.expected_form,
            length,
            container,
            behavior=self.form_mapping_info.behavior,
            buffer_key=self.form_mapping_info.buffer_key,
        )

    def mock(self):
        return awkward.typetracer.typetracer_from_form(
            self.expected_form,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
        )

    def mock_empty(self, backend="cpu"):
        return awkward.to_backend(
            self.expected_form.length_zero_array(highlevel=False),
            backend,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
        )

    def prepare_for_projection(self):
        from dask_awkward.lib.utils import trace_form_structure

        meta, report = awkward.typetracer.typetracer_with_report(
            self.expected_form,
            highlevel=True,
            behavior=self.form_mapping_info.behavior,
            buffer_key=self.form_mapping_info.buffer_key,
        )
        return (
            meta,
            report,
            {
                "trace": trace_form_structure(
                    self.expected_form,
                    buffer_key=self.form_mapping_info.buffer_key,
                ),
                "form_info": self.form_mapping_info,
            },
        )

    def necessary_columns(self, report, state):
        from dask_awkward.lib.utils import buffer_keys_required_to_compute_shapes

        form_key_to_parent_form_key = state["trace"]["form_key_to_parent_form_key"]
        form_key_to_buffer_keys = state["trace"]["form_key_to_buffer_keys"]
        form_info = state["form_info"]

        data_buffers = {
            *report.data_touched,
            *buffer_keys_required_to_compute_shapes(
                form_info.parse_buffer_key,
                report.shape_touched,
                form_key_to_parent_form_key,
                form_key_to_buffer_keys,
            ),
        }
        return frozenset(form_info.keys_for_buffer_keys(data_buffers)) & frozenset(
            self.common_keys
        )

    def project_keys(self, keys):
        return type(self)(
            base_form=self.base_form,
            expected_form=self.expected_form,
            form_mapping_info=self.form_mapping_info,
            descriptor_bytes=self.descriptor_bytes,
            descriptor_for_columns=self.descriptor_for_columns,
            common_keys=keys,
            client_kwargs=self.client_kwargs,
            call_options=self.call_options,
        )

    def project(self, report, state):
        if not self.use_optimization:
            return self
        return self.project_keys(self.necessary_columns(report, state))

    def project_manually(self, columns):
        return self.project_keys(frozenset(columns) & frozenset(self.common_keys))


def flight_dask(
    location,
    descriptor,
    *,
    form_mapping,
    descriptor_for_columns=None,
    client_kwargs=None,
    call_options=None,
):
    """Build a dask-awkward collection from an Apache Flight endpoint.

    One partition is created per ``FlightEndpoint`` returned by
    ``GetFlightInfo`` (run once at graph-construction time). Endpoints that
    advertise their own ``locations`` are honored (a fresh client connects to
    that location inside the partition read); otherwise the factory's
    ``location`` is reused.

    Parameters
    ----------
        location : str
            The Flight endpoint URI (``grpc://host:port``).
        descriptor : pyarrow.flight.FlightDescriptor
            The normalized descriptor identifying the dataset.
        form_mapping : ImplementsFormMapping
            A ``_map_schema_flight`` callable mapping the raw form to the
            NanoEvents form and returning the ``form_mapping_info`` state.
        descriptor_for_columns : Callable or None, optional
            Projection hook rebuilding a descriptor for a projected column list.
        client_kwargs : dict or None, optional
            Keyword arguments for constructing Flight clients.
        call_options : pyarrow.flight.FlightCallOptions or None, optional
            Call options passed to Flight RPCs.

    Returns
    -------
        dask_awkward.Array
            The lazy events array.
    """
    import dask_awkward

    import pyarrow.flight as flight

    client = flight.FlightClient(location, **(client_kwargs or {}))
    info = client.get_flight_info(descriptor, call_options)

    base_form = awkward.from_arrow_schema(info.schema)
    expected_form, form_mapping_info = form_mapping(base_form)

    partitions = []
    for index, endpoint in enumerate(info.endpoints):
        if endpoint.locations:
            ep_location = _location_uri(endpoint.locations[0])
        else:
            ep_location = location
        partitions.append((ep_location, endpoint.ticket.serialize(), index))

    io_fn = FromFlightFn(
        base_form=base_form,
        expected_form=expected_form,
        form_mapping_info=form_mapping_info,
        descriptor_bytes=descriptor.serialize(),
        descriptor_for_columns=descriptor_for_columns,
        client_kwargs=client_kwargs,
        call_options=call_options,
    )

    return dask_awkward.from_map(io_fn, partitions, label="from-flight")
