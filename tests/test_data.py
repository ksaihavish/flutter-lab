"""Dataset intake: format checks and the dimensional-parameter round trip."""

import json
from pathlib import Path

import pytest

from flutter_ai.data import check_dataset, params_from_metadata
from flutter_ai.flutter import flutter_analysis
from flutter_ai.params import get_preset

TEMPLATE = Path(__file__).resolve().parents[1] / "data" / "template_synthetic"


def test_template_dataset_is_valid():
    assert check_dataset(TEMPLATE).ok


def test_dimensional_parameters_round_trip_to_same_flutter_speed():
    meta = json.loads((TEMPLATE / "metadata.json").read_text())
    p = params_from_metadata(meta)
    ref = flutter_analysis(get_preset("uav_flap")).flutter_speed
    assert flutter_analysis(p).flutter_speed == pytest.approx(ref, rel=1e-6)


def test_missing_parameters_and_bad_units_are_reported(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps({"name": "x", "rig": {"b": 0.1}}))
    (tmp_path / "runs" / "r.csv").write_text("t,U,h,alpha\n0,10,0.001,5\n0.001,10,0.001,6\n")
    rep = check_dataset(tmp_path)
    assert not rep.ok
    assert any("rig parameters missing" in e for e in rep.errors)
    assert any("degrees" in w for w in rep.warnings)
