import os

from coffea.util import describe_root_file, extract_cms_provenance

SAMPLES = os.path.join(os.path.dirname(__file__), "samples")
PFNANO = os.path.join(SAMPLES, "pfnano.root")  # full CMS provenance trees
NANO_DY = os.path.join(SAMPLES, "nano_dy.root")  # Events + Runs, no ParameterSets


def test_describe_root_file():
    desc = describe_root_file(PFNANO)
    # provenance + data trees are all reported
    assert {"Events", "Runs", "ParameterSets", "MetaData"} <= set(desc)
    assert desc["Events"]["num_entries"] == 10
    assert "Jet_pt" in desc["Events"]["branches"]
    # a non-tree object (the NanoAOD 'tag' TObjString) has no branches
    assert desc["tag"]["branches"] is None


def test_extract_cms_provenance_curated():
    prov = extract_cms_provenance(PFNANO)
    # global tag chain (RECO/MiniAOD/NanoAOD steps)
    assert "106X_mc2017_realistic_v6" in prov["global_tag"]
    # processing chain and JEC provenance are surfaced
    assert {"RECO", "PAT", "NANO"} <= set(prov["process_names"])
    assert "L2Relative" in prov["jec_levels"]
    assert any(a.startswith("AK4") for a in prov["jet_algorithms"])


def test_extract_cms_provenance_full_superset_of_curated():
    full = extract_cms_provenance(PFNANO, detail="full")
    # "full" exposes the raw parameter keys; curated is a renamed subset
    assert "globaltag" in full and "@process_name" in full
    assert full["globaltag"] == extract_cms_provenance(PFNANO)["global_tag"]
    assert len(full) > 50  # the whole config string table, not just a few keys


def test_extract_cms_provenance_no_parametersets():
    assert extract_cms_provenance(NANO_DY) is None


def test_extract_cms_provenance_bad_detail():
    import pytest

    with pytest.raises(ValueError, match="curated.*full"):
        extract_cms_provenance(PFNANO, detail="everything")
