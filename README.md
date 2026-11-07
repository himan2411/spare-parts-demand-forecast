# Spare-parts demand forecasting

Forecasting how many units of each spare part a service center will need over the part's supplier lead time, and checking whether the resulting stock would cover it. Everything here runs on **synthetic** data. I have built similar forecasting systems in production. This repo shows the method on invented data.

- [METHOD.md](METHOD.md): the problem, approach, how it runs in production, and how it is evaluated.
- `parts_demand.py`: the method end to end. It generates intermittent demand per center and part, trains one gradient-boosted model across all series, backtests it against Croston-SBA and a 52-week average, and checks each center for drift.

## Run it

Needs Python 3.12 or newer (tested on 3.12).

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python parts_demand.py              # about a minute on a laptop
python parts_demand.py --no-change  # the same world without the injected change
python test_parts_demand.py         # checks: PSI, Croston-SBA, stock rule, equal stock, no look-ahead, drift
```

## Results on synthetic data

These numbers come from invented data. The run is seeded, so you get the same output.

```
SYNTHETIC data: 720 center x part series, 208 weeks, lead times 2-8 weeks
Center x part weeks with zero demand: 82%

Backtest on 8 matched windows, stock at the Poisson 95% quantile:
     method   mae   bias  fill_rate  cycle_service  avg_stock
      model 1.719 -0.119      0.871          0.878      4.614
croston_sba 1.760 -0.195      0.841          0.853      4.319
    avg_52w 1.767 -0.165      0.842          0.852      4.392

Lowest MAE by window: model 6, avg_52w 2 (of 8).
Model closer than both baselines on 2259 / 5760 series-windows.

Bias by window:
 cutoff_week  model  croston_sba  avg_52w
         171 -0.040       -0.071   -0.044
         175 -0.153       -0.166   -0.148
         179 -0.275       -0.282   -0.258
         183 -0.287       -0.290   -0.264
         187 -0.108       -0.199   -0.174
         191 -0.045       -0.194   -0.158
         195 -0.002       -0.162   -0.126
         199 -0.009       -0.156   -0.114

At equal stock (baselines scaled with hindsight to the model's average stock):
     method  scale  fill_rate  cycle_service  avg_stock
      model  1.000      0.871          0.878      4.614
croston_sba  1.097      0.860          0.868      4.614
    avg_52w  1.067      0.855          0.862      4.615

Drift per center, reference period vs last 13 weeks:
 center  fleet_change  share_ref  share_now psi_demand        flagged_by  changed (synthetic truth)
      0         0.520      0.155      0.282     0.1203 fleet, share, psi                       True
      1         0.481      0.165      0.313     0.1322 fleet, share, psi                       True
      2         0.538      0.084      0.190     0.0999      fleet, share                       True
      3         0.000      0.162      0.162     0.0002                 -                      False
      4         0.000      0.122      0.138     0.0073                 -                      False
      5         0.000      0.485      0.500     0.0022                 -                      False
      6         0.000      0.193      0.212     0.0025                 -                      False
      7         0.500      0.168      0.299     0.0979      fleet, share                       True
      8         0.000      0.158      0.168     0.0011                 -                      False
      9         0.000      0.149      0.149     0.0000                 -                      False
     10         0.000      0.126      0.136     0.0012                 -                      False
     11         0.000      0.154      0.151     0.0032                 -                      False

Demand PSI with all centers pooled: 0.015 (stable).
```

What I read from it:

- Demand is intermittent. 82% of center and part weeks have none, and 59% of the weeks that do have demand need more than one unit (up to 25).
- The model has the lowest error. Its MAE is 2.3% below Croston-SBA's, and it has the lowest MAE in 6 of 8 windows, though it is closer than both baselines on only 39% of series-windows.
- Most of its fill-rate lead is extra stock. As run, it fills 87.1% of demand against 84.1% for Croston-SBA, but holds 7% more stock. At equal stock its lead shrinks to 1.1 points (87.1% against 86.0%).
- The under-forecast is the injected change. At cutoffs 179 and 183 all three methods sit at -26% to -29% bias, even though the fleet register already shows the larger fleet by 183. The model then recovers to under 1% by cutoff 195, while the baselines stay at -11% to -16%. With `--no-change`, bias is -0.1% to -3.3% for every method, yet cycle service still misses the 95% target by 3.7 to 5.1 points (89.9% to 91.3%), because demand is lumpier than the Poisson rule assumes. The next rule to try is a negative binomial quantile or the empirical lead-time distribution.
- Centers 0, 1, 2 and 7 take on older machines from week 180. The fleet register flags all four (+48% to +54%), and so does the share of weeks with demand, which rises by 78% to 127%. Nothing else is flagged, and with `--no-change` nothing is flagged at all. The register only sees machines added or removed. The share of weeks with demand is the signal that would catch the same machines breaking down more often. Demand PSI flags only 2 of the 4, and the other two sit just under 0.1: deciles of a mostly-zero series collapse onto zero. Pooled over all centers it reads 0.015, so check per center.

## License

MIT. See [LICENSE](LICENSE).
