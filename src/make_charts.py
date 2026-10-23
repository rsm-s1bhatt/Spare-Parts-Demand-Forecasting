"""Charts for the README: forecast error by model, and the service vs. stock trade-off."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
s = pd.read_csv(R / "summary.csv")
p = s[s.policy == "Poisson"].sort_values("mean_mase")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].barh(p.model, p.mean_mase, color=["#2f6f4f" if m == "TSB" else "#9aa5a0" for m in p.model])
ax[0].invert_yaxis(); ax[0].set_xlabel("Mean MASE, 12-month holdout (lower is better)")
ax[0].set_title("Forecast error, 2,509 spare parts")
for y, v in enumerate(p.mean_mase): ax[0].text(v + 0.01, y, f"{v:.2f}", va="center")
ax[0].set_xlim(0, 1.6)

for pol, mk in [("Poisson", "o"), ("NegBin", "s")]:
    q = s[(s.policy == pol) & (s.model != "Naive")]
    ax[1].scatter(q.total_avg_on_hand, q.fill_rate * 100, marker=mk, s=60, label=f"{pol} stocking")
    for _, r in q.iterrows():
        ax[1].annotate(r.model, (r.total_avg_on_hand, r.fill_rate * 100), fontsize=8,
                       xytext=(4, 3), textcoords="offset points")
ax[1].axhline(95, ls="--", c="grey", lw=1); ax[1].text(5800, 95.3, "95% target", fontsize=8, color="grey")
ax[1].set_xlabel("Average units on hand, all parts"); ax[1].set_ylabel("Fill rate, %")
ax[1].set_title("Service vs. stock (Naive omitted: 38% fill)"); ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(R / "results.png", dpi=150)
