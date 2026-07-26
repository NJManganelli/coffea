"""Conversions between the pydantic dataset specs and ServiceX.

Two directions, mirroring the dual dict/model contract used elsewhere in
``dataset_tools``:

* **ServiceX ``deliver()`` output -> DataGroupSpec** (:func:`from_servicex`).
  Wraps the ``dict[name -> delivered files]`` that ``servicex.deliver`` returns
  into a :class:`~coffea.dataset_tools.filespec.DataGroupSpec` of *input* files
  (run :func:`~coffea.dataset_tools.preprocess.preprocess` afterwards to fill in
  steps / uuid / form). This direction needs **no** ServiceX dependency.

* **DataGroupSpec / DatasetSpec -> ServiceX** as either the class objects
  (:func:`to_servicex_spec`, :func:`to_servicex_sample`) or the dependency-free
  spec dict (:func:`to_servicex_dict`).

Every ``import servicex`` is function-local, so importing this module (and hence
``coffea.dataset_tools``) never requires ServiceX to be installed; only the
class-producing helpers do.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from coffea.dataset_tools.filespec import (
    DataGroupSpec,
    DatasetSpec,
    identify_file_format,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import servicex

# ServiceX writes its ROOT deliveries to a TTree; "servicex" is the usual
# default but the name depends on the codegen, hence the override.
DEFAULT_TREE = "servicex"


# --------------------------------------------------------------------------- #
# ServiceX deliver() output  ->  DataGroupSpec   (dependency-free)
# --------------------------------------------------------------------------- #
def _iter_delivered(sample_result: Any) -> Iterable[str]:
    """Yield file-path strings from one ``deliver()`` sample value.

    The container type varies by ServiceX version: older clients expose a
    ``.files`` attribute or a plain list, while 3.3.x returns a ``GuardList``
    (a Sequence, not a list subclass, with no ``.files``). Iterating a *failed*
    GuardList re-raises the underlying transform exception, surfacing the error.
    """
    if hasattr(sample_result, "files"):
        yield from (str(f) for f in sample_result.files)
    elif isinstance(sample_result, list):
        yield from (str(f) for f in sample_result)
    else:
        for f in sample_result:
            yield str(f)


def datasetspec_from_delivered(
    files: Iterable[str],
    *,
    object_path: str = DEFAULT_TREE,
    metadata: Mapping[Any, Any] | None = None,
) -> DatasetSpec:
    """Wrap delivered file paths into a (not-yet-preprocessed) DatasetSpec.

    ROOT files receive *object_path* as their tree; Parquet files (which forbid
    an object path) receive ``None``. The result carries only file locations --
    run ``preprocess`` to populate steps, entry counts, uuids and the form.
    """
    file_map: dict[str, dict[str, str] | None] = {}
    for f in files:
        f = str(f)
        if identify_file_format(f) == "root":
            file_map[f] = {"object_path": object_path}
        else:
            file_map[f] = None
    return DatasetSpec(files=file_map, metadata=dict(metadata or {}))


def from_servicex(
    delivered: Mapping[str, Any],
    *,
    object_path: str = DEFAULT_TREE,
    metadata: Mapping[Any, Any] | None = None,
) -> DataGroupSpec:
    """Convert a ``servicex.deliver()`` result into a DataGroupSpec.

    Parameters
    ----------
    delivered : Mapping[str, Any]
        The ``dict[sample_name -> delivered files]`` returned by
        ``servicex.deliver`` (values may be ``GuardList``, ``list[str]`` or an
        object with ``.files``).
    object_path : str
        TTree name assigned to delivered ROOT files (default ``"servicex"``).
    metadata : Mapping, optional
        Metadata applied to *every* resulting dataset.

    Returns
    -------
    DataGroupSpec
        One dataset per sample, holding input files only. Feed it to
        ``preprocess`` to obtain a fully-known fileset.
    """
    if not isinstance(delivered, Mapping):
        raise TypeError(
            "from_servicex expects a mapping of sample_name -> delivered files "
            f"(the servicex.deliver output), got {type(delivered)}"
        )
    group: dict[str, DatasetSpec] = {}
    for name, sample_result in delivered.items():
        paths = list(_iter_delivered(sample_result))
        if not paths:
            raise ValueError(f"sample {name!r} delivered no files")
        group[name] = datasetspec_from_delivered(
            paths, object_path=object_path, metadata=metadata
        )
    return DataGroupSpec(group)


# --------------------------------------------------------------------------- #
# DatasetSpec / DataGroupSpec  ->  ServiceX classes   (lazy servicex import)
# --------------------------------------------------------------------------- #
def _servicex_dataset(spec: DatasetSpec, *, prefer_did: bool):
    """Build a ServiceX dataset identifier for a DatasetSpec."""
    from servicex import dataset

    if prefer_did and spec.did:
        return dataset.Rucio(spec.did)
    return dataset.FileList([str(k) for k in spec.files.keys()])


def to_servicex_sample(
    spec: DatasetSpec,
    *,
    name: str,
    query: Any = None,
    codegen: str | None = None,
    num_files: int | None = None,
    prefer_did: bool = True,
) -> servicex.Sample:
    """Build a ``servicex.Sample`` from a DatasetSpec.

    Uses the dataset's ``did`` (as ``dataset.Rucio``) when *prefer_did* and a did
    is present, otherwise a ``dataset.FileList`` of the spec's file keys. *query*
    (a ServiceX query object) and *codegen* are passed through when supplied.
    """
    from servicex import Sample

    kwargs: dict[str, Any] = {
        "Name": name,
        "Dataset": _servicex_dataset(spec, prefer_did=prefer_did),
    }
    if query is not None:
        kwargs["Query"] = query
    if codegen is not None:
        kwargs["Codegen"] = codegen
    if num_files is not None:
        kwargs["NFiles"] = num_files
    return Sample(**kwargs)


def to_servicex_spec(
    group: DataGroupSpec,
    *,
    query: Any = None,
    codegen: str | None = None,
    num_files: int | None = None,
    general: Any = None,
    prefer_did: bool = True,
) -> servicex.ServiceXSpec:
    """Build a ``servicex.ServiceXSpec`` (one Sample per dataset) from a group.

    *query* is applied to every sample; build the spec per dataset with
    :func:`to_servicex_sample` if you need distinct queries. *general* is an
    optional ``servicex.General`` passed through unchanged.
    """
    from servicex import ServiceXSpec

    samples = [
        to_servicex_sample(
            spec,
            name=name,
            query=query,
            codegen=codegen,
            num_files=num_files,
            prefer_did=prefer_did,
        )
        for name, spec in group.items()
    ]
    kwargs: dict[str, Any] = {"Sample": samples}
    if general is not None:
        kwargs["General"] = general
    return ServiceXSpec(**kwargs)


# --------------------------------------------------------------------------- #
# DataGroupSpec  ->  ServiceX spec dict   (dependency-free)
# --------------------------------------------------------------------------- #
def _sample_dict(
    name: str,
    spec: DatasetSpec,
    *,
    query: Any,
    codegen: str | None,
    num_files: int | None,
    prefer_did: bool,
) -> dict[str, Any]:
    sample: dict[str, Any] = {"Name": name}
    if prefer_did and spec.did:
        sample["RucioDID"] = spec.did
    else:
        files = [str(k) for k in spec.files.keys()]
        if files and all(f.startswith("root://") for f in files):
            sample["XRootDFiles"] = files
        else:
            raise ValueError(
                f"dataset {name!r}: a local FileList cannot be expressed in the "
                "dependency-free ServiceX dict (ServiceX needs a dataset.FileList "
                "object). Provide a Rucio did, use xrootd (root://) URLs, or use "
                "to_servicex_spec()/to_servicex_sample() for the class-based form."
            )
    if query is not None:
        sample["Query"] = query
    if codegen is not None:
        sample["Codegen"] = codegen
    if num_files is not None:
        sample["NFiles"] = num_files
    return sample


def to_servicex_dict(
    group: DataGroupSpec,
    *,
    query: Any = None,
    codegen: str | None = None,
    num_files: int | None = None,
    general: Mapping[str, Any] | None = None,
    prefer_did: bool = True,
) -> dict[str, Any]:
    """Build a dependency-free ServiceX spec dict from a DataGroupSpec.

    Each dataset becomes a Sample keyed by its ``did`` (``RucioDID``) or, failing
    that, its xrootd URLs (``XRootDFiles``) -- the two forms ``servicex.deliver``
    accepts from a plain mapping without constructing ``dataset.*`` objects. A
    local file list raises (use the class form). The returned dict can be passed
    straight to ``servicex.deliver``.

    Note: recent ServiceX deprecates the ``RucioDID`` / ``XRootDFiles`` sample
    fields in favour of ``Dataset=dataset.Rucio(...)`` / ``dataset.FileList(...)``
    (still functional). Prefer :func:`to_servicex_spec` when a ServiceX dependency
    is acceptable.
    """
    samples = [
        _sample_dict(
            name,
            spec,
            query=query,
            codegen=codegen,
            num_files=num_files,
            prefer_did=prefer_did,
        )
        for name, spec in group.items()
    ]
    out: dict[str, Any] = {"Sample": samples}
    if general is not None:
        out["General"] = dict(general)
    return out
