# Method

## The problem

A service organisation keeps spare parts at regional service centers. Each center looks after a fleet of machines at customer sites. When a machine breaks down, the technician needs the part from the shelf now. If it is not there, the machine stays down until a reorder arrives, which takes the part's supplier lead time: two to eight weeks here.

So the question for each center and part is not "how much next week". It is "how much over the next lead time, and how much stock covers that with high probability". Three things make that hard:

- **Demand is intermittent.** For most center and part pairs, most weeks see no demand at all.
- **Demand is lumpy.** A breakdown can need several units at once, so a non-zero week is often more than one unit.
- **The fleet changes.** When a center takes on more or older machines, breakdowns rise. The fleet register shows the extra machines, but not that older ones fail more often.

## The approach

1. **Forecast lead-time demand directly.** One row per series and forecast origin. The target is total demand over that part's lead time, which is the quantity stock has to cover.
2. **One global model, not one per series.** A gradient-boosted tree model trained across every center and part, with Poisson loss for counts. Sparse series borrow strength from similar ones.
3. **Features built for intermittent demand.** Recent and yearly average demand, the share of weeks with any demand, the average size of a non-zero week, weeks since the last demand, the Croston-SBA rate, the lead time, and the fleet size from the register. Only weeks up to the forecast origin are read. The tests fail if a feature reads a later week, or if the backtest trains on a lead time that ends after its cutoff.
4. **A stocking rule on top.** Each forecast becomes a stock level: the Poisson 95% quantile of the forecast lead-time demand. The same rule applies to every method, so the comparison is about the forecasts.
5. **Drift checks per center.** The reference period (weeks 104-155) against the last 13 weeks, one center at a time, on three signals: the fleet register (flag a change beyond 10%), the share of weeks with demand (flag a change beyond 25%), and the population stability index of weekly demand (flag at 0.1). The register sees machines added or removed. Only the demand signals can see the same machines breaking down more often.

## How it runs in production

A weekly job does five things:

1. Reads demand history from service work orders, the fleet register, and the part master with its lead times.
2. Builds the features for every center and part as of the end of the week.
3. Scores the model and turns each forecast into a reorder point with the stocking rule.
4. Hands the reorder points to the replenishment system. It orders when stock on hand plus stock on order falls below the reorder point.
5. Runs the per-center drift check and the rolling backtest. A flagged center, or a fill rate falling short of target, gets a person to look before the model is retrained.

The script in this repo is the modelling core of that job, run locally with pandas and scikit-learn.

## Evaluation

- **Rolling origin.** At each cutoff, train only on lead times that ended by then. Forecast every series' next lead time and compare with what was demanded. Move the cutoff 4 weeks and repeat, 8 times.
- **Matched windows.** Every method is scored on the same series and the same weeks.
- **Baselines a parts planner would use.** Croston's method with the Syntetos-Boylan correction, which is the standard for intermittent demand, and a plain 52-week average. Both rates are multiplied by the lead time.
- **Error and service side by side.** MAE and bias for the forecast. Fill rate (share of demand the stock would have covered), cycle service (share of lead times with no shortfall) and average stock for the decision. A forecast that only looks good on error but needs more stock for the same fill is not better.
- **Equal stock.** A method can buy fill with stock. So each baseline is also scaled up or down until its average stock matches the model's, and fill is compared at that level. The scale is picked with hindsight, so this shows the trade-off between stock and fill, not a baseline you could deploy.

## The synthetic data

`make_data()` invents 4 years of weekly demand for 12 centers and 60 parts. Breakdowns are Poisson, scaled by each center's fleet and each part's failure rate. Each breakdown needs one unit plus a Poisson number of extras. Each part gets a lead time of 2, 4, 6 or 8 weeks. From week 180, a third of the centers take on older machines: the fleet grows 50% and each machine breaks down 50% more often. Run `python parts_demand.py --no-change` for the same world without that change.
