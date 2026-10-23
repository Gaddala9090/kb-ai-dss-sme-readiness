"""Property tests for the knowledge base and the hybrid model.  Run: python -m pytest -q"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from kbdss import CONTROLS, DIMENSIONS, INDICATORS, FuzzyKnowledgeBase, KBAIDSS, recommend, trimf


def firms(levels: dict | None = None, n: int = 1, default: int = 3) -> pd.DataFrame:
    row = {c: default for c in INDICATORS} | {"size_log": 0.0, "firm_age": 10,
                                               "sector_ict": 0.0, "sector_services": 1.0}
    for d, v in (levels or {}).items():
        row |= {c: v for c in DIMENSIONS[d]}
    return pd.DataFrame([row] * n)


def test_membership_shoulders_and_partition():
    x = np.linspace(1, 5, 41)
    assert trimf(1, 1, 1, 3) == 1 and trimf(5, 3, 5, 5) == 1
    total = trimf(x, 1, 1, 3) + trimf(x, 2, 3, 4) + trimf(x, 3, 5, 5)
    assert (total > 0).all()                                  # every input is covered


def test_rule_base_size_and_bounds():
    assert len(FuzzyKnowledgeBase.rules()) == 54
    r = FuzzyKnowledgeBase().infer(pd.concat([firms(default=1), firms(default=5)]))["readiness"]
    assert 0 <= r.iloc[0] < 25 and 75 < r.iloc[1] <= 100


def test_readiness_is_monotone_in_every_indicator():
    rng = np.random.default_rng(0)
    base = firms(n=200)
    base[INDICATORS] = rng.integers(1, 5, (200, len(INDICATORS)))
    kb = FuzzyKnowledgeBase()
    r0 = kb.infer(base)["readiness"].values
    for c in INDICATORS:
        up = base.copy(); up[c] = np.minimum(up[c] + 1, 5)
        assert (kb.infer(up)["readiness"].values >= r0 - 1e-9).all(), c


def test_technology_bottleneck_caps_readiness():
    weak_tech = firms({"data_infrastructure": 1, "digital_skills": 5}, default=5)
    assert FuzzyKnowledgeBase().infer(weak_tech)["readiness"].iloc[0] < 50


def test_hybrid_is_sklearn_compatible_and_recommends():
    rng = np.random.default_rng(1)
    X = firms(n=120)
    X[INDICATORS] = rng.integers(1, 6, (120, len(INDICATORS)))
    y = (X[DIMENSIONS["data_infrastructure"]].mean(axis=1) + rng.normal(0, 1, 120) > 3).astype(int)
    m = clone(KBAIDSS(learner="rf")).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert 0 <= m.weight_ <= 1 and ((p >= 0) & (p <= 1)).all()
    rec = recommend(m, X.iloc[[0]])
    assert len(rec) == len(DIMENSIONS) + 3 and set(CONTROLS) <= set(X.columns)
