"""Operating-envelope study (envelope.py): the pieces, without the full sweep."""

from flutter_ai.envelope import _highest_clean, _run, small_signal
from flutter_ai.experiments import neural_available
from flutter_ai.flutter import flutter_speed
from flutter_ai.params import get_preset


def test_highest_clean_ratio_stops_at_first_failure():
    rows = [{"ratio": r, "failures": f} for r, f in [(0.9, 3), (1.0, 0), (1.1, 0), (1.2, 2), (1.3, 0)]]
    assert _highest_clean(rows) == 1.1  # below-flutter rows are ignored; 1.3 does not count after 1.2 failed
    assert _highest_clean([{"ratio": 1.0, "failures": 1}]) is None


def test_small_signal_matches_design_expectations():
    p = get_preset("uav_flap")
    ss = small_signal(p, flutter_speed(p), ratios=(0.9, 1.1))
    assert ss["0.90"]["open_loop"]["spectral_radius"] < 1.0 < ss["1.10"]["open_loop"]["spectral_radius"]
    assert ss["1.10"]["lqr"]["spectral_radius"] < 1.0  # the design point: the linear loop is stable
    if neural_available(p):
        assert ss["1.10"]["lqg"]["spectral_radius"] < 1.0 and ss["1.10"]["neural"]["spectral_radius"] < 1.0


def test_unlimited_servo_run_reports_rate_demand():
    p = get_preset("uav_flap")
    Uf = flutter_speed(p)
    real = _run(("uav_flap", "discrete_gust", 1.1, "lqr", False, 0, Uf))
    free = _run(("uav_flap", "discrete_gust", 1.1, "lqr", True, 0, Uf))
    assert not real["failed"] and not free["failed"]
    assert free["rate_p99_deg_s"] > 0 and 0.0 <= free["rate_above_limit"] <= 1.0
