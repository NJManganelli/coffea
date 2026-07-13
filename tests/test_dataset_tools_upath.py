"""Tests for upath / file-handle support on the pydantic dataset specs."""

import copy

import pytest

from coffea.dataset_tools.filespec import DatasetSpec


class _FakeUPath:
    """Stand-in for a universal_pathlib.UPath key carrying storage_options."""

    def __init__(self, s, **so):
        self._s = s
        self._so = so

    def __str__(self):
        return self._s

    @property
    def storage_options(self):
        return self._so


@pytest.fixture
def local_root(tmp_path):
    p = tmp_path / "a.root"
    p.write_bytes(b"root-bytes")
    return str(p)


# --------------------------------------------------------------------------- #
# UPath key canonicalization + storage_options capture
# --------------------------------------------------------------------------- #
def test_upath_key_is_canonicalized_and_storage_options_captured():
    ds = DatasetSpec(
        files={
            _FakeUPath("root://x//a.root", token="SECRET", endpoint="http://e"): {
                "object_path": "Events"
            }
        }
    )
    (key,) = list(ds.files.keys())
    assert key == "root://x//a.root"  # canonical string, not the object
    assert ds.files[key].storage_options == {"token": "SECRET", "endpoint": "http://e"}


def test_explicit_storage_options_are_not_overwritten_by_key():
    ds = DatasetSpec(
        files={
            _FakeUPath("root://x//a.root", token="FROM_KEY"): {
                "object_path": "Events",
                "storage_options": {"token": "EXPLICIT"},
            }
        }
    )
    (fs,) = list(ds.files.values())
    assert fs.storage_options == {"token": "EXPLICIT"}


# --------------------------------------------------------------------------- #
# storage_options / handles are never serialized; load does not auto-open
# --------------------------------------------------------------------------- #
def test_storage_options_excluded_from_serialization():
    ds = DatasetSpec(files={"root://x//a.root": {"object_path": "Events"}})
    next(iter(ds.files.values())).storage_options = {"token": "SECRET"}

    dumped = ds.model_dump_json()
    assert "SECRET" not in dumped
    assert "storage_options" not in dumped
    assert "storage_options" not in ds.model_dump()


def test_reload_has_no_credentials_and_no_open_handle(local_root):
    ds = DatasetSpec(files={local_root: {"object_path": "Events"}})
    next(iter(ds.files.values())).storage_options = {"token": "SECRET"}
    ds.open(local_root)  # a live handle exists on the original

    reloaded = DatasetSpec.model_validate_json(ds.model_dump_json())
    fs = next(iter(reloaded.files.values()))
    assert fs.storage_options is None  # credentials not persisted
    assert fs.is_open is False  # loading never opens a handle
    ds.close()


# --------------------------------------------------------------------------- #
# open / close / context manager
# --------------------------------------------------------------------------- #
def test_open_close_roundtrip(local_root):
    ds = DatasetSpec(files={local_root: {"object_path": "Events"}})
    fs = ds.files[local_root]
    assert fs.is_open is False

    handle = ds.open(local_root)
    assert fs.is_open is True
    assert handle.read() == b"root-bytes"

    ds.close(local_root)
    assert fs.is_open is False
    ds.close()  # idempotent, closes all (none open)


def test_opened_context_manager_closes_on_exit(local_root):
    ds = DatasetSpec(files={local_root: {"object_path": "Events"}})
    with ds.opened() as handles:
        assert set(handles) == {local_root}
        assert ds.files[local_root].is_open is True
    assert ds.files[local_root].is_open is False


def test_deepcopy_with_open_handle_does_not_raise(local_root):
    # DataGroupSpec.limit_steps deep-copies specs; an open handle must not break it.
    ds = DatasetSpec(files={local_root: {"object_path": "Events"}})
    ds.open(local_root)
    try:
        copy.deepcopy(ds)
    finally:
        ds.close()


# --------------------------------------------------------------------------- #
# upath(): configured UPath for uproot, requires the optional dependency
# --------------------------------------------------------------------------- #
def test_upath_requires_universal_pathlib_or_returns_configured_path():
    ds = DatasetSpec(files={"root://x//a.root": {"object_path": "Events"}})
    fs = next(iter(ds.files.values()))
    fs.storage_options = {"token": "T"}
    upath = pytest.importorskip("upath")

    p = ds.upath("root://x//a.root", endpoint="http://e")
    assert isinstance(p, upath.UPath)
    # storage_options are merged (spec + overrides)
    assert p.storage_options.get("token") == "T"
    assert p.storage_options.get("endpoint") == "http://e"


def test_upath_import_error_without_dependency(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "upath", None)
    ds = DatasetSpec(files={"root://x//a.root": {"object_path": "Events"}})
    with pytest.raises(ImportError):
        ds.upath("root://x//a.root")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
