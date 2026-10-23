"""Reproduces every figure, table and number reported in the paper.

    python run.py                         # synthetic benchmark (seed 42), about 10 minutes
    python run.py --data my_survey.csv    # your own SME survey with the same columns

Outputs (results/): fig1_architecture.png, fig2_learning_curve.png, figS1_shap.png,
table1_knowledge_base.csv, table2_performance.csv, metrics.json, rule_base.txt, sme_dataset.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kbdss import (CONTROLS, DIMENSIONS, INDICATORS, KNOWLEDGE_FEATURES, MODULES,
                   FuzzyKnowledgeBase, FuzzyOnly, KBAIDSS, recommend)

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
SEED = 42
PROPOSED = "KB-AI-DSS (proposed)"


# --------------------------------------------------------------------------- #
# Synthetic SME benchmark                                                      #
# --------------------------------------------------------------------------- #
def simulate_smes(n: int = 800, seed: int = SEED) -> pd.DataFrame:
    """Survey-like SME data. The generating process encodes stylised facts from the
    digital-transformation literature (a common digital-orientation factor, size and
    sector effects, complementarity between data infrastructure and skills, diminishing
    returns) and is written independently of the fuzzy rule base."""
    rng = np.random.default_rng(seed)
    employees = rng.integers(10, 250, n)
    size = (np.log(employees) - np.log(employees).mean()) / np.log(employees).std()
    age = rng.integers(1, 40, n)
    sector = rng.choice(["manufacturing", "services", "ict"], n, p=[0.4, 0.4, 0.2])
    ict = (sector == "ict").astype(float)
    g = rng.normal(0, 1, n)
    shift = {"data_infrastructure": 0.25 * size + 0.5 * ict, "digital_skills": 0.5 * ict,
             "processes": 0.3 * size, "leadership": -0.01 * (age - 20)}
    q = {d: 0.6 * g + 0.8 * rng.normal(0, 1, n) + shift.get(d, 0) for d in DIMENSIONS}
    df = pd.DataFrame({"employees": employees, "size_log": size, "firm_age": age,
                       "sector_ict": ict, "sector_services": (sector == "services").astype(float)})
    for d, cols in DIMENSIONS.items():
        for c in cols:
            df[c] = np.clip(np.rint(3 + 0.9 * q[d] + rng.normal(0, 0.6, n)), 1, 5).astype(int)
    tech = np.minimum(q["data_infrastructure"], q["digital_skills"])
    eta = (-0.4 + 1.2 * np.tanh(tech) + 0.6 * np.tanh(q["innovation_culture"])
           + 0.45 * q["strategy"] + 0.3 * q["leadership"] + 0.25 * q["processes"]
           + 0.5 * (q["strategy"] > 0) * (q["innovation_culture"] > 0)
           + 0.2 * size + 0.3 * ict - 0.01 * (age - 20))
    df["innovation_potential"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
class Select(TransformerMixin, BaseEstimator):
    """Gives every baseline exactly the same raw inputs (indicators + controls)."""
    def __init__(self, cols=tuple(INDICATORS + CONTROLS)): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X[list(self.cols)].values


def models() -> dict:
    return {
        "Fuzzy knowledge base only": FuzzyOnly(),
        "Logistic regression": make_pipeline(Select(), StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": make_pipeline(Select(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=1)),
        "Gradient boosting": KBAIDSS(use_knowledge=False, blend=False),
        "Ablation: feature fusion only": KBAIDSS(use_knowledge=True, blend=False, learner="rf"),
        "Ablation: blend with GB learner": KBAIDSS(learner="gb"),
        PROPOSED: KBAIDSS(learner="rf"),
    }


def _fmt_p(p: float) -> str:
    return "—" if np.isnan(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def benchmark(df, y):
    """Full-sample evaluation: 10-fold stratified CV repeated 5 times (50 folds)."""
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
    s = {n: cross_validate(m, df, y, cv=cv, n_jobs=-1, scoring=("accuracy", "f1", "roc_auc"))
         for n, m in models().items()}
    ref = s[PROPOSED]["test_roc_auc"]
    rows = [{"Model": n,
             "Accuracy": f"{v['test_accuracy'].mean():.3f} ± {v['test_accuracy'].std():.3f}",
             "F1-score": f"{v['test_f1'].mean():.3f} ± {v['test_f1'].std():.3f}",
             "AUC": f"{v['test_roc_auc'].mean():.3f} ± {v['test_roc_auc'].std():.3f}",
             "p (AUC vs proposed)": _fmt_p(np.nan if n == PROPOSED else wilcoxon(ref, v["test_roc_auc"]).pvalue)}
            for n, v in s.items()]
    raw = {n: {k[5:]: float(v[k].mean()) for k in v if k.startswith("test_")} for n, v in s.items()}
    return pd.DataFrame(rows), raw


def learning_curve(df, y, sizes=(40, 80, 160, 320, 560), repeats=20):
    """Small-sample regime typical of SME surveys: train on n firms, test on 240 held out."""
    def one(n, rep, tr, te):
        out = []
        for name, m in models().items():
            mm = clone(m).fit(df.iloc[tr], y[tr])
            out.append({"n": n, "rep": rep, "model": name,
                        "auc": roc_auc_score(y[te], mm.predict_proba(df.iloc[te])[:, 1]),
                        "w": getattr(mm, "weight_", np.nan)})
        return out
    jobs = [(n, rep, tr, te) for n in sizes for rep, (tr, te) in enumerate(
        StratifiedShuffleSplit(repeats, train_size=n, test_size=240, random_state=1).split(df, y))]
    rows = Parallel(n_jobs=-1)(delayed(one)(*j) for j in jobs)
    return pd.DataFrame([r for batch in rows for r in batch])


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "savefig.dpi": 300})
INK, ACCENT, QUIET, GREY = "#1f2937", "#1d4ed8", "#4b5563", "#9ca3af"


def _box(ax, x, y, w, h, title, body, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#dbeafe" if accent else "white", ec=ACCENT if accent else INK,
                                lw=1.5 if accent else 0.9))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", weight="bold", color=INK, fontsize=7.1)
    ax.text(x + w / 2, y + h - 0.58, body, ha="center", va="top", color=QUIET, fontsize=7,
            linespacing=1.4)


def _arrow(ax, a, b, label=None, dx=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=0.9, color=INK))
    if label:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, fontsize=6.5, color=QUIET,
                ha="center", style="italic")


def figure_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.9, 3.5))
    ax.set(xlim=(0, 16.3), ylim=(0, 6.25)); ax.axis("off")
    _box(ax, 0.1, 2.0, 2.3, 2.1, "SME data", "18 Likert indicators\n(6 dimensions × 3)\nsize, age, sector")
    _box(ax, 3.0, 3.7, 3.0, 2.3, "Fuzzy knowledge base",
         "3 capability modules\n× 9 rules, top module\n27 rules (54 in total)\nmonotone inference")
    _box(ax, 6.6, 3.9, 3.0, 1.9, "Knowledge model $p_K$", "logistic link on\n3 capabilities\n+ readiness score")
    _box(ax, 6.6, 0.2, 3.0, 2.3, "Data-driven model $p_D$",
         "random forest on\nindicators, controls\nand knowledge\noutputs")
    _box(ax, 10.1, 2.1, 3.3, 1.95, "Knowledge-guided blend",
         "$p = w\\,p_K + (1-w)\\,p_D$\nw learned by\ninner cross-validation", accent=True)
    _box(ax, 13.9, 3.9, 2.35, 1.9, "Explanation", "rule trace,\nlinguistic labels,\nSHAP attributions")
    _box(ax, 13.9, 0.3, 2.35, 1.9, "Recommendation", "what-if ranking\nof dimensions\nand modules")
    ax.text(4.5, 6.15, "rules encoded from digital-maturity literature", ha="center", fontsize=6.5,
            color=QUIET, style="italic")
    _arrow(ax, (2.4, 3.6), (3.0, 4.6))
    _arrow(ax, (2.4, 2.5), (6.6, 1.15), "raw inputs", dx=-0.4, dy=-0.42)
    _arrow(ax, (6.0, 4.85), (6.6, 4.85))
    _arrow(ax, (4.5, 3.7), (6.6, 1.75), "knowledge outputs", dx=-0.85, dy=-0.05)
    _arrow(ax, (9.6, 4.4), (10.1, 3.25)); _arrow(ax, (9.6, 1.6), (10.1, 2.5))
    _arrow(ax, (13.4, 3.25), (13.9, 4.4)); _arrow(ax, (13.4, 2.5), (13.9, 1.6))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


CURVE_STYLE = {"Fuzzy knowledge base only": dict(color=INK, ls="--", marker="s"),
               "Logistic regression": dict(color=GREY, ls=":", marker="^"),
               "Gradient boosting": dict(color=GREY, ls="-.", marker="v"),
               "Random forest": dict(color="#6b7280", ls="-", marker="o"),
               PROPOSED: dict(color=ACCENT, ls="-", marker="D", lw=2.2)}


def figure_learning_curve(lc: pd.DataFrame, path: Path) -> pd.DataFrame:
    agg = lc.groupby(["model", "n"]).agg(auc=("auc", "mean"), sd=("auc", "std"),
                                         k=("auc", "size"), w=("w", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for name, style in CURVE_STYLE.items():
        style = dict(style)
        d = agg[agg.model == name]
        ci = 1.96 * d.sd / np.sqrt(d.k)
        ax.plot(d.n, d.auc, label=name, ms=4.5, lw=style.pop("lw", 1.3), **style)
        if name == PROPOSED:
            ax.fill_between(d.n, d.auc - ci, d.auc + ci, color=ACCENT, alpha=0.12, lw=0)
            for n, w in zip(d.n, d.w):
                ax.text(n, 0.035, f"w = {w:.2f}", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=6.8, color=ACCENT)
            ax.text(0.015, 0.09, "learned knowledge weight", transform=ax.transAxes,
                    fontsize=6.5, color=ACCENT, style="italic")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(lc.n.unique())); ax.set_xticklabels(sorted(lc.n.unique()))
    ax.set(xlabel="Training sample size (firms)", ylabel="Test AUC (240 held-out firms)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    lo = agg[agg.model.isin(CURVE_STYLE)].auc.min()
    ax.set_ylim(lo - 0.03, None)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return agg


def figure_shap(model: KBAIDSS, df: pd.DataFrame, path: Path) -> pd.Series:
    """Supplementary: SHAP attributions of the data-driven component."""
    Xt = model.transform(df)
    sv = np.asarray(shap.TreeExplainer(model.model_).shap_values(Xt))
    if sv.ndim == 3:                                     # (n, features, classes)
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns).sort_values(ascending=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.barh([f.replace("_", " ") for f in top.index], top.values,
            color=[ACCENT if f in KNOWLEDGE_FEATURES else GREY for f in top.index])
    ax.set_xlabel("Mean |SHAP value| (probability of high innovation potential)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(color=ACCENT, label="Knowledge-derived feature"),
                       Patch(color=GREY, label="Raw indicator or control")],
              frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return imp


def table_knowledge_base() -> pd.DataFrame:
    return pd.DataFrame([{"Capability module": cap.replace("_cap", "").title(),
                          "Dimension": d.replace("_", " ").capitalize(),
                          "Indicators (Likert 1–5)": ", ".join(DIMENSIONS[d]),
                          "Aggregation logic": logic.capitalize(), "Rules": 9}
                         for cap, (a, b, logic) in MODULES.items() for d in (a, b)])


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV with indicator, control and innovation_potential columns")
    ap.add_argument("--quick", action="store_true", help="fewer repeats, for a smoke test")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(args.data) if args.data else simulate_smes()
    y = df["innovation_potential"].values
    df.to_csv(OUT / "sme_dataset.csv", index=False)

    table_knowledge_base().to_csv(OUT / "table1_knowledge_base.csv", index=False)
    (OUT / "rule_base.txt").write_text("\n".join(FuzzyKnowledgeBase.rules()) + "\n")
    figure_architecture(OUT / "fig1_architecture.png")

    perf, raw = benchmark(df, y)
    perf.to_csv(OUT / "table2_performance.csv", index=False)
    lc = learning_curve(df, y, repeats=4 if args.quick else 20)
    curve = figure_learning_curve(lc, OUT / "fig2_learning_curve.png")

    small = {}
    for n in sorted(lc.n.unique()):
        d = lc[lc.n == n].pivot(index="rep", columns="model", values="auc")
        small[int(n)] = {"auc": d.mean().round(4).to_dict(),
                         "p_vs_random_forest": float(wilcoxon(d[PROPOSED], d["Random forest"]).pvalue),
                         "p_vs_gradient_boosting": float(wilcoxon(d[PROPOSED], d["Gradient boosting"]).pvalue),
                         "mean_weight_w": float(lc[(lc.n == n) & (lc.model == PROPOSED)].w.mean())}

    final = KBAIDSS(learner="rf").fit(df, y)
    imp = figure_shap(final, df, OUT / "figS1_shap.png")
    kb = FuzzyKnowledgeBase().infer(df)
    case = int((kb["readiness"] - kb["readiness"].quantile(0.25)).abs().idxmin())
    rec = recommend(final, df.loc[[case]])

    metrics = {
        "n_firms": int(len(df)), "positive_rate": float(y.mean()),
        "n_rules": len(FuzzyKnowledgeBase.rules()),
        "full_sample_cv": "10-fold stratified, 5 repeats (50 folds)", "full_sample": raw,
        "learning_curve": small,
        "readiness": {"mean": float(kb.readiness.mean()), "sd": float(kb.readiness.std()),
                      "corr_with_outcome": float(np.corrcoef(kb.readiness, y)[0, 1])},
        "final_weight_w": final.weight_,
        "shap_top10": imp.head(10).round(4).to_dict(),
        "knowledge_share_of_shap": float(imp[KNOWLEDGE_FEATURES].sum() / imp.sum()),
        "case_firm": {"index": case, "readiness": rec.attrs["readiness"],
                      "probability": rec.attrs["probability"],
                      "dimension_scores": kb.loc[case, list(DIMENSIONS)].round(2).to_dict(),
                      "ranking": rec.round(4).to_dict(orient="records")},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(perf.to_string(index=False))
    print(curve.pivot(index="n", columns="model", values="auc").round(3).to_string())
    print(json.dumps({k: metrics[k] for k in ("learning_curve", "final_weight_w", "readiness",
                                              "knowledge_share_of_shap", "case_firm")}, indent=2))


if __name__ == "__main__":
    main()
