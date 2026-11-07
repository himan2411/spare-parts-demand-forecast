"""Checks for parts_demand.py. Run: python test_parts_demand.py"""

import numpy as np
import pandas as pd

import parts_demand as pdm


def test_psi():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 10, 5000)
    assert pdm.psi(base, rng.normal(100, 10, 5000)) < 0.1
    assert pdm.psi(base, rng.normal(115, 10, 5000)) > 0.25


def test_croston_sba_on_a_regular_series():
    u = np.zeros((1, 104), int)
    u[0, 3::4] = 3  # 3 units every 4th week
    assert np.isclose(pdm.croston_sba(u)[0], (1 - pdm.ALPHA / 2) * 3 / 4)


def test_stock_rule():
    assert pdm.stock(np.array([0.0, 2.0])).tolist() == [0, 5]  # Poisson(2): P(X <= 5) = 0.983


def test_no_future_demand_in_features():
    d = pdm.make_data(centers=2, parts=6)
    origin = 100
    before = pdm.features(d, origin)
    d["demand"][:, origin + 1 :] = 10**6  # scramble every week after the origin
    d["fleet"][:, origin + 1 :] = 10**6
    after = pdm.features(d, origin)
    cols = [c for c in before.columns if c != "y"]
    assert before[cols].equals(after[cols])


def test_target_is_demand_over_the_next_lead_time():
    d = pdm.make_data(centers=2, parts=6)
    d["demand"] = np.arange(d["demand"].size).reshape(d["demand"].shape)  # every week distinct and non-zero
    origin = 100
    y = pdm.features(d, origin).y.to_numpy()
    want = [d["demand"][i, origin + 1 : origin + 1 + lead].sum() for i, lead in enumerate(d["lead"])]
    assert (y == want).all(), (y, want)


def test_training_rows_end_by_cutoff():
    d = pdm.make_data(centers=2, parts=6)
    rows = pdm.training_rows(d, 120)
    assert rows.origin.min() >= 52
    assert rows.end.max() <= 120


def test_short_history_is_refused():
    d = pdm.make_data(centers=2, parts=6)
    for call in (
        lambda: pdm.features(d, 40),
        lambda: pdm.training_rows(d, 50),
        lambda: pdm.drift({**d, "demand": d["demand"][:, :150]}),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected a refusal for less than a year of history")


def test_backtest_trains_only_up_to_each_cutoff():
    seen, real_fit = [], pdm.fit

    def spy(train):
        seen.append(int(train.end.max()))
        return real_fit(train)

    pdm.fit = spy
    try:
        bt = pdm.backtest(pdm.make_data(centers=3, parts=10), windows=2)
    finally:
        pdm.fit = real_fit
    cutoffs = sorted(bt.cutoff_week.unique().tolist())
    assert len(seen) == len(cutoffs) and all(s <= c for s, c in zip(seen, cutoffs, strict=True)), (seen, cutoffs)


def test_drift_flags_the_changed_centers_and_nothing_else():
    for change in (False, True):
        d = pdm.make_data(change=change)
        flags = pdm.drift(d).flagged_by
        assert d["changed"].any() == change
        assert ((flags != "-") == d["changed"]).all(), (change, flags.tolist())
        # The demand signal must see the change on its own, not only the fleet register.
        assert (flags.str.contains("share") == d["changed"]).all(), (change, flags.tolist())


def test_drift_flags_a_shrinking_fleet():
    d = pdm.make_data(change=False)
    d["fleet"][d["center"] == 0, -pdm.RECENT :] *= 0.5
    assert "fleet" in pdm.drift(d).flagged_by[0]


def test_equal_stock_matches_the_model_stock():
    y = np.random.default_rng(0).poisson(3, 1000)
    bt = pd.DataFrame({"y": y, "model": 3.0, "croston_sba": 4.0, "avg_52w": 1.5})
    out = pdm.equal_stock(bt)
    assert (out.avg_stock == pdm.stock(np.array([3.0]))[0]).all(), out
    assert out.scale.iloc[1] < 1 < out.scale.iloc[2], out  # one scaled down, one up
    bt["avg_52w"] = 0.1  # would need about 30x: refuse rather than print unequal stock
    try:
        pdm.equal_stock(bt)
    except ValueError:
        return
    raise AssertionError("expected a refusal when no scale matches the stock")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
