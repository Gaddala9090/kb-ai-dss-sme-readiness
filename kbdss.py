"""KB-AI-DSS: a knowledge-based AI decision support system for assessing digital
transformation readiness and innovation potential in entrepreneurial SMEs.

Three layers, all in this module:
  1. FuzzyKnowledgeBase  - hierarchical fuzzy rule-based inference over six readiness
                           dimensions (54 rules) -> monotone, transparent readiness 0-100.
  2. KBAIDSS             - scikit-learn classifier that blends a knowledge-calibrated model
                           with a data-driven learner (weight learned by inner CV) to
                           predict innovation potential.
  3. recommend()         - what-if engine that ranks which dimension to improve first.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Knowledge model: dimensions, indicators (Likert 1-5) and capability modules  #
# --------------------------------------------------------------------------- #
DIMENSIONS: dict[str, list[str]] = {
    "strategy": ["str_vision", "str_roadmap", "str_budget"],
    "leadership": ["lead_commitment", "lead_change", "lead_risk"],
    "data_infrastructure": ["data_cloud", "data_integration", "data_quality"],
    "digital_skills": ["skill_analytics", "skill_ai", "skill_training"],
    "processes": ["proc_automation", "proc_digitized", "proc_customer"],
    "innovation_culture": ["cult_experiment", "cult_openness", "cult_learning"],
}
INDICATORS = [i for v in DIMENSIONS.values() for i in v]
CONTROLS = ["size_log", "firm_age", "sector_ict", "sector_services"]

# capability module -> (input A, input B, aggregation logic)
MODULES = {
    "strategic_cap": ("strategy", "leadership", "compensatory"),
    "technological_cap": ("data_infrastructure", "digital_skills", "bottleneck"),
    "organizational_cap": ("processes", "innovation_culture", "compensatory"),
}
KNOWLEDGE_FEATURES = list(MODULES) + ["readiness"]

L, M, H = 0, 1, 2
LEVELS = ("Low", "Medium", "High")
READINESS_TERMS = ("Very low", "Low", "Medium", "High", "Very high")

# Sub-module consequents (row = level of input A, column = level of input B).
# compensatory: strengths offset weaknesses; bottleneck: the weaker input dominates
# (technology adoption needs both infrastructure and the skills to use it).
CONSEQUENTS = {
    "compensatory": [[L, L, M], [L, M, H], [M, H, H]],
    "bottleneck": [[L, L, L], [L, M, M], [L, M, H]],
}


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership; a == b or b == c gives a shoulder."""
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b > a else (x >= a).astype(float)
    right = (c - x) / (c - b) if c > b else (x <= c).astype(float)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


# Ruspini partitions (memberships sum to 1 everywhere) and consequent term centres.
IN_MF = [(1, 1, 3), (1, 3, 5), (3, 5, 5)]                       # Likert 1-5: Low/Medium/High
CAP_MF, CAP_C = [(0, 0, 5), (0, 5, 10), (5, 10, 10)], np.array([0.0, 5.0, 10.0])
RDY_C = np.array([0.0, 25.0, 50.0, 75.0, 100.0])               # Very low ... Very high


def _infer(strengths: list[np.ndarray], terms: list[int], centres: np.ndarray) -> np.ndarray:
    """Product inference with weighted-average (height) defuzzification. On Ruspini
    partitions with a monotone rule table this is a multilinear interpolation of the
    table, so the output is provably non-decreasing in every input."""
    w = np.stack(strengths)                                          # (rules, n)
    return (w * centres[terms][:, None]).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)


def _top_consequent(i: int, j: int, k: int) -> int:
    """Top-level rule: additive evidence, but weak technology caps readiness at Low."""
    term = [0, 1, 1, 2, 3, 3, 4][i + j + k]
    return min(term, 1) if j == L else term


class FuzzyKnowledgeBase:
    """Hierarchical fuzzy system: 3 capability modules x 9 rules + 27 top rules = 54."""

    def dimension_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({d: X[c].mean(axis=1) for d, c in DIMENSIONS.items()},
                            index=X.index)

    @staticmethod
    def _memberships(x: np.ndarray, mfs) -> list[np.ndarray]:
        return [trimf(x, *mf) for mf in mfs]

    def infer(self, X: pd.DataFrame) -> pd.DataFrame:
        dims = self.dimension_scores(X)
        out, cap_mu = {}, {}
        for cap, (a, b, logic) in MODULES.items():
            mu_a, mu_b = self._memberships(dims[a].values, IN_MF), self._memberships(dims[b].values, IN_MF)
            ws, ts = [], []
            for i, j in itertools.product(range(3), range(3)):
                ws.append(mu_a[i] * mu_b[j])
                ts.append(CONSEQUENTS[logic][i][j])
            out[cap] = _infer(ws, ts, CAP_C)
            cap_mu[cap] = self._memberships(out[cap], CAP_MF)
        s, t, o = (cap_mu[c] for c in MODULES)
        ws, ts = [], []
        for i, j, k in itertools.product(range(3), repeat=3):
            ws.append(s[i] * t[j] * o[k])
            ts.append(_top_consequent(i, j, k))
        out["readiness"] = _infer(ws, ts, RDY_C)
        return pd.concat([dims, pd.DataFrame(out, index=X.index)], axis=1)

    @staticmethod
    def rules() -> list[str]:
        """Human-readable rule base (the system's explicit knowledge)."""
        lines = []
        for cap, (a, b, logic) in MODULES.items():
            for i, j in itertools.product(range(3), range(3)):
                lines.append(f"IF {a} is {LEVELS[i]} AND {b} is {LEVELS[j]} "
                             f"THEN {cap} is {LEVELS[CONSEQUENTS[logic][i][j]]}")
        for i, j, k in itertools.product(range(3), repeat=3):
            lines.append(f"IF strategic_cap is {LEVELS[i]} AND technological_cap is {LEVELS[j]} "
                         f"AND organizational_cap is {LEVELS[k]} "
                         f"THEN readiness is {READINESS_TERMS[_top_consequent(i, j, k)]}")
        return lines


# --------------------------------------------------------------------------- #
# Hybrid model and baselines (scikit-learn compatible)                         #
# --------------------------------------------------------------------------- #
class KBAIDSS(ClassifierMixin, BaseEstimator):
    """Knowledge-guided hybrid classifier.

    p(x) = w * p_K(x) + (1 - w) * p_D(x)
      p_K : knowledge-calibrated model (logistic link on the 4 fuzzy outputs)
      p_D : gradient boosting or random forest (learner) on indicators + controls (+ fuzzy outputs if use_knowledge)
      w   : chosen on the training data only, by inner cross-validated log-loss.
    The rule base acts as a prior that dominates when data are scarce and yields to
    the data-driven model as evidence accumulates. blend=False gives feature fusion only.
    """

    def __init__(self, use_knowledge: bool = True, blend: bool = True, learner: str = "gb",
                 n_estimators: int = 200, learning_rate: float = 0.03, max_depth: int = 2,
                 random_state: int = 0):
        self.use_knowledge = use_knowledge
        self.blend = blend
        self.learner = learner
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        base = X[INDICATORS + CONTROLS]
        K = FuzzyKnowledgeBase().infer(X)[KNOWLEDGE_FEATURES] if (self.use_knowledge or self.blend) else None
        return (pd.concat([base, K], axis=1) if self.use_knowledge else base), K

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._features(X)[0]

    def _gb(self):
        if self.learner == "rf":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                          random_state=self.random_state, n_jobs=1)
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, subsample=0.8, min_samples_leaf=10,
            random_state=self.random_state)

    def fit(self, X, y):
        y = np.asarray(y)
        Xd, K = self._features(X)
        self.model_ = self._gb()
        self.weight_ = 0.0
        if self.blend:
            self.kb_model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            k = int(min(5, np.bincount(y).min()))
            cv = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
            p_d = cross_val_predict(self._gb(), Xd, y, cv=cv, method="predict_proba")[:, 1]
            p_k = cross_val_predict(self.kb_model_, K, y, cv=cv, method="predict_proba")[:, 1]
            grid = np.linspace(0, 1, 21)
            losses = [log_loss(y, np.clip(w * p_k + (1 - w) * p_d, 1e-6, 1 - 1e-6)) for w in grid]
            self.weight_ = float(grid[int(np.argmin(losses))])
            self.kb_model_.fit(K, y)
        self.model_.fit(Xd, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        Xd, K = self._features(X)
        p = self.model_.predict_proba(Xd)[:, 1]
        if self.blend:
            p = self.weight_ * self.kb_model_.predict_proba(K)[:, 1] + (1 - self.weight_) * p
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class FuzzyOnly(ClassifierMixin, BaseEstimator):
    """Knowledge-only baseline: readiness score with an F1-optimal training threshold."""

    def fit(self, X, y):
        r = FuzzyKnowledgeBase().infer(X)["readiness"].values
        grid = np.linspace(r.min(), r.max(), 101)
        self.threshold_ = grid[np.argmax([f1_score(y, r >= g, zero_division=0) for g in grid])]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = FuzzyKnowledgeBase().infer(X)["readiness"].values / 100.0
        return np.column_stack([1 - p, p])

    def predict(self, X):
        r = FuzzyKnowledgeBase().infer(X)["readiness"].values
        return (r >= self.threshold_).astype(int)


# --------------------------------------------------------------------------- #
# Recommendation engine                                                        #
# --------------------------------------------------------------------------- #
def recommend(model: KBAIDSS, firm: pd.DataFrame, step: float = 1.0) -> pd.DataFrame:
    """What-if analysis: raise the indicators of each single dimension, and of each
    capability module (both of its dimensions), by `step` (max 5); report the change in
    fuzzy readiness and in predicted innovation probability. Module-level actions matter
    because a bottleneck module cannot be lifted by improving one of its inputs alone."""
    kb = FuzzyKnowledgeBase()
    base_r = kb.infer(firm)["readiness"].iloc[0]
    base_p = model.predict_proba(firm)[0, 1]
    actions = {d: [d] for d in DIMENSIONS}
    actions.update({f"{m} (module)": [a, b] for m, (a, b, _) in MODULES.items()})
    rows = []
    for action, dims in actions.items():
        cols = [c for d in dims for c in DIMENSIONS[d]]
        what_if = firm.copy()
        what_if[cols] = np.minimum(what_if[cols] + step, 5)
        rows.append({"action": action, "n_dimensions": len(dims),
                     "current_score": float(firm[cols].mean(axis=1).iloc[0]),
                     "readiness_gain": float(kb.infer(what_if)["readiness"].iloc[0] - base_r),
                     "probability_gain": float(model.predict_proba(what_if)[0, 1] - base_p)})
    out = pd.DataFrame(rows).sort_values("probability_gain", ascending=False)
    out.attrs.update(readiness=float(base_r), probability=float(base_p))
    return out.reset_index(drop=True)
