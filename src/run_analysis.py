"""Intermittent spare-parts demand: classification, forecasting backtest, and stocking test.

Data: Hyndman's `carparts` dataset (monthly demand for 2,674 car spare parts,
Jan 1998 - Mar 2002; from the R package expsmooth). Series with missing months
are dropped, leaving 2,509 complete series, the subset used in the literature.

Steps
1. Classify each part on the training window with Syntetos-Boylan-Croston
   (ADI = average inter-demand interval, CV2 = squared coefficient of variation
   of non-zero demand sizes; cut-offs 1.32 and 0.49).
2. Forecast the last 12 months from the first 39 with Naive, Historic Average,
   Croston, SBA, and TSB (statsforecast), and score each with MASE.
3. Stocking test: an order-up-to policy (monthly review, 1-month lead time,
   Poisson demand at the forecast rate) aiming for a 95% target, scored on
   achieved fill rate and average units on hand over the 12 holdout months.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from statsforecast import StatsForecast
from statsforecast.models import (
    TSB,
    CrostonClassic,
    CrostonSBA,
    HistoricAverage,
    Naive,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "carparts_wide.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

H = 12              # holdout months
TARGET = 0.95       # service-level target for the stocking test
LEAD = 1            # months between ordering and receiving
ADI_CUT, CV2_CUT = 1.32, 0.49


def classify(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for part in train.columns:
        y = train[part].to_numpy()
        nz = y[y > 0]
        n_nz = len(nz)
        adi = len(y) / n_nz if n_nz else np.inf
        cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if n_nz > 1 else 0.0
        if n_nz == 0:
            cls = "No demand"
        elif adi < ADI_CUT and cv2 < CV2_CUT:
            cls = "Smooth"
        elif adi >= ADI_CUT and cv2 < CV2_CUT:
            cls = "Intermittent"
        elif adi < ADI_CUT:
            cls = "Erratic"
        else:
            cls = "Lumpy"
        rows.append({"part": part, "adi": adi, "cv2": cv2,
                     "nonzero_months": n_nz, "class": cls})
    return pd.DataFrame(rows)


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    ds = pd.date_range("1998-01-01", periods=len(wide), freq="MS")
    long = wide.assign(ds=ds).melt(id_vars="ds", var_name="unique_id", value_name="y")
    return long[["unique_id", "ds", "y"]]


def mase(actual, fc, train):
    scale = np.mean(np.abs(np.diff(train)))
    if scale == 0:
        return np.nan  # constant training history; MASE undefined
    return np.mean(np.abs(actual - fc)) / scale


def order_up_to(rate, disp=1.0):
    """S = 95% quantile of demand over lead time + review period.
    disp = variance / mean of monthly demand. disp <= 1 -> Poisson; > 1 ->
    negative binomial with the same mean and the inflated variance."""
    mu = rate * (LEAD + 1)
    if mu <= 0:
        return 0
    if disp <= 1:
        return poisson.ppf(TARGET, mu)
    p = 1 / disp                      # nbinom: mean = n(1-p)/p, var = mean/p
    n = mu * p / (1 - p)
    return nbinom.ppf(TARGET, n, p)


def simulate_stock(demand, S):
    """Order-up-to policy. Each month: receive last month's order, serve demand
    (lost sales), then order back up to S."""
    on_hand, pipeline = S, 0.0
    served = held = 0.0
    for d in demand:
        on_hand += pipeline
        sale = min(on_hand, d)
        served += sale
        on_hand -= sale
        held += on_hand
        pipeline = max(S - on_hand, 0)
    return served, demand.sum(), held / len(demand)


def main():
    wide = pd.read_csv(DATA)
    wide = wide.loc[:, wide.notna().all()]
    train, test = wide.iloc[:-H], wide.iloc[-H:]

    cls = classify(train)
    cls.to_csv(OUT / "classification.csv", index=False)

    models = [Naive(), HistoricAverage(), CrostonClassic(), CrostonSBA(),
              TSB(alpha_d=0.1, alpha_p=0.1)]
    sf = StatsForecast(models=models, freq="MS", n_jobs=-1)
    fc = sf.forecast(df=to_long(train), h=H)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    names = [c for c in fc.columns if c not in ("unique_id", "ds")]

    rows = []
    for part, g in fc.groupby("unique_id"):
        y_tr, y_te = train[part].to_numpy(), test[part].to_numpy()
        for m in names:
            f = g[m].to_numpy()
            rate = float(f.mean())
            served, demanded, oh = simulate_stock(y_te, order_up_to(rate))
            rows.append({"part": part, "model": m, "policy": "Poisson",
                         "mase": mase(y_te, f, y_tr),
                         "served": served, "demanded": demanded,
                         "avg_on_hand": oh})
            disp = y_tr.var() / y_tr.mean() if y_tr.mean() > 0 else 1.0
            served, demanded, oh = simulate_stock(y_te, order_up_to(rate, disp))
            rows.append({"part": part, "model": m, "policy": "NegBin",
                         "mase": mase(y_te, f, y_tr),
                         "served": served, "demanded": demanded,
                         "avg_on_hand": oh})
    res = pd.DataFrame(rows).merge(cls[["part", "class"]], on="part")
    res.to_csv(OUT / "per_part_results.csv", index=False)

    # Fill rate pooled by units (served / demanded), on-hand summed across parts
    summ = (res.groupby(["policy", "model"])
               .agg(mean_mase=("mase", "mean"),
                    median_mase=("mase", "median"),
                    served=("served", "sum"),
                    demanded=("demanded", "sum"),
                    total_avg_on_hand=("avg_on_hand", "sum"))
               .sort_values("mean_mase"))
    summ["fill_rate"] = summ["served"] / summ["demanded"]
    summ = summ.reset_index().sort_values(["policy", "mean_mase"])
    by_class = (res[res.policy == "Poisson"].pivot_table(index="class", columns="model",
                                values="mase", aggfunc="mean"))
    counts = cls["class"].value_counts()

    summ.to_csv(OUT / "summary.csv")
    by_class.to_csv(OUT / "mase_by_class.csv")
    counts.to_csv(OUT / "class_counts.csv")
    print("Parts:", wide.shape[1])
    print(counts.to_string(), "\n")
    print(summ.round(3).to_string(), "\n")
    print(by_class.round(3).to_string())


if __name__ == "__main__":
    main()
