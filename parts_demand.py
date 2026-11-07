"""Spare-parts demand forecasting on SYNTHETIC service-center data.

All data is invented by make_data() below and describes no real business.
Demand for a part at a service center is intermittent: most weeks nothing,
then a breakdown needs one or several units. What matters is the demand over
the part's supplier lead time, because that is what the stock on the shelf
has to cover until a reorder arrives.

The script trains one gradient-boosted model across every center and part to
forecast lead-time demand. It backtests the model against Croston-SBA and a
plain 52-week average on matched windows, scores each method by error and by
the fill rate its stock would reach, and checks each center for drift.

Run: python parts_demand.py
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 3
WEEKS = 208
LEAD_TIMES = [2, 4, 6, 8]  # weeks from reorder to delivery, one per part
CHANGE_WEEK = 180  # some centers take on older machines from this week on
WINDOWS = 8
STEP = 4  # weeks between backtest cutoffs
SERVICE = 0.95  # stock target: cover lead-time demand in 95% of reorder cycles
ALPHA = 0.1  # Croston smoothing constant
EPS = 1e-4
REF = slice(104, 156)  # reference period: weeks 104-155, a full year before the comparison
RECENT = 13  # weeks compared against the reference period
FLEET_TOL = 0.1  # flag a center whose fleet moved more than 10% from the reference period
SHARE_TOL = 0.25  # flag a center whose share of weeks with demand moved more than 25%
FEATURES = ["center", "part", "lead", "fleet", "ma13", "ma52", "share52", "size52", "since", "sba"]


def make_data(centers=12, parts=60, change=True):
    """Synthetic weekly demand per center x part. Every machine in a center's
    fleet breaks down now and then, and each breakdown needs one unit of a part
    plus a few extra units on average, so demand is mostly zero with spikes.
    With `change`, a third of the centers take on older machines from
    CHANGE_WEEK: the fleet grows 50% and each machine breaks down 50% more
    often. The fleet register records the extra machines; nothing records the
    higher breakdown rate."""
    rng = np.random.default_rng(SEED)
    center = np.repeat(np.arange(centers), parts)
    part = np.tile(np.arange(parts), centers)
    t = np.arange(WEEKS)
    changed = np.zeros(centers, bool)
    changed[rng.choice(centers, max(1, centers // 3), replace=False)] = change  # same draws either way

    after = changed[center][:, None] & (t >= CHANGE_WEEK)
    fleet = np.round(np.round(rng.lognormal(3.4, 0.4, centers))[center][:, None] * np.where(after, 1.5, 1.0))
    rate = rng.lognormal(-5.2, 0.8, parts)[part][:, None]  # breakdowns per machine per week needing the part
    extra = rng.lognormal(-0.5, 0.6, parts)[part][:, None]  # extra units per breakdown, on average
    breakdowns = rng.poisson(fleet * rate * np.where(after, 1.5, 1.0))
    demand = breakdowns + rng.poisson(extra * breakdowns)
    lead = rng.choice(LEAD_TIMES, parts)[part]
    return {"demand": demand, "fleet": fleet, "center": center, "part": part, "lead": lead, "changed": changed}


def croston_sba(u):
    """Croston's method with the Syntetos-Boylan correction, per series: smooth
    the size of non-zero demands and the gap between them, and forecast
    size / gap per week. Starts from the first 52 weeks of `u`, updates week by
    week, and returns the weekly rate forecast after the last week of `u`."""
    first = u[:, :52]
    hits = (first > 0).sum(1)
    z = first.sum(1) / np.maximum(hits, 1)
    p = 52 / np.maximum(hits, 1)
    last = 51 - np.argmax(first[:, ::-1] > 0, axis=1)
    for t in range(52, u.shape[1]):
        nz = u[:, t] > 0
        z = np.where(nz, z + ALPHA * (u[:, t] - z), z)
        p = np.where(nz, p + ALPHA * (t - last - p), p)
        last = np.where(nz, t, last)
    return (1 - ALPHA / 2) * z / p


def features(d, origin):
    """One row per series for a forecast made at the end of week `origin` of the
    demand over the following `lead` weeks. Demand features read weeks up to
    the origin only. Fleet size at the origin comes from the fleet register."""
    if origin < 52:
        raise ValueError("need a full year of history before the forecast origin")
    weeks = d["demand"].shape[1]
    u = d["demand"][:, : origin + 1]
    last52 = u[:, -52:]
    hits = (last52 > 0).sum(1)
    end = origin + d["lead"]  # last week of the lead time
    cum = np.concatenate([np.zeros((len(u), 1), int), d["demand"].cumsum(1)], axis=1)
    rows = np.arange(len(u))
    y = cum[rows, np.minimum(end, weeks - 1) + 1] - cum[:, origin + 1]
    return pd.DataFrame(
        {
            "center": d["center"],
            "part": d["part"],
            "lead": d["lead"],
            "fleet": d["fleet"][:, origin],
            "ma13": u[:, -13:].mean(1),
            "ma52": last52.mean(1),
            "share52": hits / 52,
            "size52": last52.sum(1) / np.maximum(hits, 1),
            "since": origin - np.where(u > 0, np.arange(origin + 1), -1).max(1),
            "sba": croston_sba(u),
            "y": np.where(end < weeks, y, np.nan),
            "origin": origin,
            "end": end,
        }
    )


def training_rows(d, cutoff):
    """Forecasts whose whole lead time ends at or before the cutoff. Origins are
    taken every 4 weeks back from the cutoff: neighbouring origins share most
    of their history, so this cuts run time with little loss."""
    if cutoff <= 52:
        raise ValueError("need more than a year of history before the cutoff")
    rows = pd.concat([features(d, o) for o in range(cutoff - 1, 51, -4)])
    rows = rows[rows.end <= cutoff]
    if rows.empty:
        raise ValueError("no lead time ends by the cutoff yet")
    return rows


def fit(train):
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=300,
        learning_rate=0.05,
        # Native categoricals cap center and part at 255 levels each; for a larger
        # catalogue, encode them (target or frequency) or use part attributes.
        categorical_features=[0, 1],
        early_stopping=False,  # use every training row
        random_state=SEED,
    )
    return model.fit(train[FEATURES], train["y"])


def stock(f):
    """Units to hold so lead-time demand is covered in SERVICE of reorder cycles,
    if that demand were Poisson with mean f. The same rule for every method."""
    return poisson.ppf(SERVICE, f)


METHODS = ["model", "croston_sba", "avg_52w"]


def backtest(d, windows=WINDOWS):
    """Rolling origin: for each cutoff, train only on lead times that ended by
    then, forecast every series' next lead time, and keep what was demanded.
    Returns one row per series and window, so every method is scored on the
    same series and weeks."""
    if windows < 1:
        raise ValueError("need at least one backtest window")
    last = d["demand"].shape[1] - 1 - max(LEAD_TIMES)
    out = []
    for cutoff in range(last - STEP * (windows - 1), last + 1, STEP):
        test = features(d, cutoff)
        out.append(
            pd.DataFrame(
                {
                    "cutoff_week": cutoff,
                    "y": test.y,
                    "model": fit(training_rows(d, cutoff)).predict(test[FEATURES]),
                    "croston_sba": test.sba * test.lead,
                    "avg_52w": test.ma52 * test.lead,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def score(y, f, scale=1.0):
    s = stock(f * scale)
    return {
        "mae": np.abs(y - f).mean(),
        "bias": f.sum() / y.sum() - 1,
        "fill_rate": np.minimum(y, s).sum() / y.sum(),
        "cycle_service": (y <= s).mean(),
        "avg_stock": s.mean(),
    }


def summary(bt):
    return pd.DataFrame([{"method": m, **score(bt.y, bt[m])} for m in METHODS])


def equal_stock(bt):
    """Scale each baseline up or down until its stock matches the model's on
    average, then compare fill. The scale is chosen with hindsight, so this shows the
    trade-off between stock and fill, not a baseline you could deploy."""
    target = stock(bt.model).mean()
    rows = [{"method": "model", "scale": 1.0, **score(bt.y, bt.model)}]
    for m in METHODS[1:]:
        lo, hi = 0.0, 10.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if stock(bt[m] * mid).mean() >= target else (mid, hi)
        if abs(stock(bt[m] * hi).mean() - target) > 0.01 * target:
            raise ValueError(f"{m}: no scale from 0 to 10 matches the model's stock within 1%")
        rows.append({"method": m, "scale": hi, **score(bt.y, bt[m], hi)})
    return pd.DataFrame(rows).drop(columns=["mae", "bias"])


def psi(base, cur):
    """Population stability index with baseline-decile bins."""
    cuts = np.unique(np.quantile(base, np.arange(1, 10) / 10, method="lower"))
    b = np.bincount(np.searchsorted(cuts, base, "left"), minlength=len(cuts) + 1) / len(base)
    c = np.bincount(np.searchsorted(cuts, cur, "left"), minlength=len(cuts) + 1) / len(cur)
    b, c = np.maximum(b, EPS), np.maximum(c, EPS)
    return float(((c - b) * np.log(c / b)).sum())


def level(v):
    return "stable" if v < 0.1 else "watch" if v < 0.25 else "drifted"


def drift(d):
    """Per center, the reference period against the last RECENT weeks, on three
    signals: the fleet register (relative change), the share of weeks with
    demand (relative change), and PSI of weekly demand. Checked per center,
    because a change in a few centers is diluted when every center is pooled.
    The register only sees machines added or removed; the demand signals are
    what would catch the same machines breaking down more often."""
    if d["demand"].shape[1] - RECENT < REF.stop:
        raise ValueError("the recent weeks overlap the reference period; need more history")
    out = []
    for c in np.unique(d["center"]):
        rows = d["center"] == c
        u, f = d["demand"][rows], d["fleet"][rows]
        share_ref, share_now = (u[:, REF] > 0).mean(), (u[:, -RECENT:] > 0).mean()
        out.append(
            {
                "center": c,
                "fleet_change": f[:, -RECENT:].mean() / f[:, REF].mean() - 1,
                "share_ref": share_ref,
                "share_now": share_now,
                "psi_demand": psi(u[:, REF].ravel(), u[:, -RECENT:].ravel()),
            }
        )
    out = pd.DataFrame(out)
    flags = pd.DataFrame(
        {
            "fleet": out.fleet_change.abs() > FLEET_TOL,
            "share": (out.share_now / out.share_ref - 1).abs() > SHARE_TOL,
            "psi": out.psi_demand.map(level) != "stable",
        }
    )
    out["flagged_by"] = flags.apply(lambda r: ", ".join(k for k, v in r.items() if v) or "-", axis=1)
    return out


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--no-change"]):
        sys.exit("usage: python parts_demand.py [--no-change]")
    change = sys.argv[1:] != ["--no-change"]
    d = make_data(change=change)
    n, w = d["demand"].shape
    print(f"SYNTHETIC data{'' if change else ', no change injected'}: {n} center x part series, {w} weeks, ", end="")
    print(f"lead times {min(LEAD_TIMES)}-{max(LEAD_TIMES)} weeks")
    print(f"Center x part weeks with zero demand: {(d['demand'] == 0).mean():.0%}\n")

    bt = backtest(d)
    k = bt.cutoff_week.nunique()
    err = bt[METHODS].sub(bt.y, axis=0).abs()
    wins = (err.model < err[["croston_sba", "avg_52w"]].min(axis=1)).sum()
    by_window = bt.groupby("cutoff_week")
    lowest = err.groupby(bt.cutoff_week).mean().idxmin(axis=1).value_counts()
    print(f"Backtest on {k} matched windows, stock at the Poisson {SERVICE:.0%} quantile:")
    print(summary(bt).to_string(index=False, float_format="{:.3f}".format))
    print(f"\nLowest MAE by window: {', '.join(f'{m} {c}' for m, c in lowest.items())} (of {k}).")
    print(f"Model closer than both baselines on {wins} / {len(bt)} series-windows.")
    print("\nBias by window:")
    bias = by_window[METHODS].sum().div(by_window.y.sum(), axis=0) - 1
    print(bias.reset_index().to_string(index=False, float_format="{:.3f}".format))
    print("\nAt equal stock (baselines scaled with hindsight to the model's average stock):")
    print(equal_stock(bt).to_string(index=False, float_format="{:.3f}".format))

    dr = drift(d)
    dr["changed (synthetic truth)"] = d["changed"]
    pooled = psi(d["demand"][:, REF].ravel(), d["demand"][:, -RECENT:].ravel())
    print(f"\nDrift per center, reference period vs last {RECENT} weeks:")
    print(dr.to_string(index=False, float_format="{:.3f}".format, formatters={"psi_demand": "{:.4f}".format}))
    print(f"\nDemand PSI with all centers pooled: {pooled:.3f} ({level(pooled)}).")
