# =============================================================================
# core/models/regime.py  —  HMM + GMM Ensemble with BMA
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import N_REGIMES, HMM_N_RESTARTS, HMM_N_ITER, GMM_N_INIT, TRAIN_RATIO


class RegimeDetector:
    """
    Ensemble: GaussianHMM + GMM with Bayesian Model Averaging.

    P(St=k | data) = w_HMM · P_HMM(St=k | X1:t) + w_GMM · P_GMM(St=k | Xt)

    BMA weights from held-out validation log-likelihood:
        wm ∝ exp(mean per-sample log-likelihood on val set)
    """

    LABELS = ["Bull", "Sideways", "Bear"]

    def __init__(self, n_regimes=N_REGIMES, n_restarts=HMM_N_RESTARTS,
                 train_ratio=TRAIN_RATIO, seed=42):
        self.n_regimes   = n_regimes
        self.n_restarts  = n_restarts
        self.train_ratio = train_ratio
        self.seed        = seed
        self.scaler      = None
        self.hmm_        = None
        self.gmm_        = None
        self.w_hmm       = 0.5
        self.w_gmm       = 0.5
        self.state_map   : dict[int, str] = {}
        self.fitted      = False
        self._X_scaled   = None
        self._dates      = None

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, r21d: pd.Series) -> "RegimeDetector":
        arr  = X.values.astype(float)
        n    = len(arr)
        n_tr = int(n * self.train_ratio)

        self.scaler = StandardScaler()
        Xtr  = self.scaler.fit_transform(arr[:n_tr])
        Xval = self.scaler.transform(arr[n_tr:])
        Xall = self.scaler.transform(arr)

        # Fit models
        self.hmm_ = self._fit_hmm(Xtr)
        self.gmm_ = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full",
            n_init=GMM_N_INIT, max_iter=500, random_state=self.seed
        ).fit(Xtr)

        # BMA weights
        self.w_hmm, self.w_gmm = self._bma(Xval)

        # Label states by mean 21d return
        states = self.hmm_.predict(Xall)
        self.state_map = self._label(states, r21d.reindex(X.index).values)

        self._X_scaled = Xall
        self._dates    = X.index
        self.fitted    = True
        return self

    def predict_proba(self, X: "pd.DataFrame | None" = None) -> pd.DataFrame:
        assert self.fitted, "fit() first"
        if X is not None:
            Xs = self.scaler.transform(X.values.astype(float))
            idx = X.index
        else:
            Xs = self._X_scaled
            idx = self._dates

        try:
            p_hmm = self.hmm_.predict_proba(Xs)
        except Exception:
            p_hmm = np.full((len(Xs), self.n_regimes), 1/self.n_regimes)
        try:
            p_gmm = self.gmm_.predict_proba(Xs)
        except Exception:
            p_gmm = np.full((len(Xs), self.n_regimes), 1/self.n_regimes)

        ens = self.w_hmm * p_hmm + self.w_gmm * p_gmm

        df = pd.DataFrame(0.0, index=idx, columns=self.LABELS)
        for int_k, name in self.state_map.items():
            if int_k < ens.shape[1]:
                df[name] += ens[:, int_k]

        row_s = df.sum(axis=1).replace(0, 1)
        df    = df.div(row_s, axis=0)
        df["regime"] = df[self.LABELS].idxmax(axis=1)
        return df

    def info(self) -> dict:
        d = {"w_hmm": round(self.w_hmm,4), "w_gmm": round(self.w_gmm,4),
             "state_map": self.state_map}
        if self.hmm_:
            d["transmat"]     = np.round(self.hmm_.transmat_, 4).tolist()
            d["means_shape"]  = list(self.hmm_.means_.shape)
            try:
                d["log_likelihood"] = float(self.hmm_.monitor_.history[-1])
            except Exception:
                pass
        return d

    # ── Private ───────────────────────────────────────────────────────────────

    def _fit_hmm(self, X: np.ndarray) -> hmm.GaussianHMM:
        best, best_s = None, -np.inf
        for seed in range(self.n_restarts):
            for cov in ("full", "diag"):
                try:
                    m = hmm.GaussianHMM(n_components=self.n_regimes,
                                        covariance_type=cov,
                                        n_iter=HMM_N_ITER, tol=1e-4,
                                        random_state=seed, verbose=False)
                    m.fit(X)
                    s = m.score(X)
                    if s > best_s:
                        best_s, best = s, m
                except Exception:
                    continue
        if best is None:
            best = hmm.GaussianHMM(n_components=self.n_regimes,
                                   covariance_type="diag", n_iter=100,
                                   random_state=0).fit(X)
        return best

    def _bma(self, Xval: np.ndarray) -> tuple[float, float]:
        if len(Xval) == 0:
            return 0.5, 0.5
        try:
            ll_hmm = self.hmm_.score(Xval) / len(Xval)
        except Exception:
            ll_hmm = -1e6
        try:
            ll_gmm = float(np.mean(self.gmm_.score_samples(Xval)))
        except Exception:
            ll_gmm = -1e6
        lls = np.clip(np.array([ll_hmm, ll_gmm], float), -1e4, 0)
        lls -= lls.max()
        w = np.exp(lls); w /= w.sum()
        return float(w[0]), float(w[1])

    def _label(self, states: np.ndarray, returns: np.ndarray) -> dict[int,str]:
        mean_r = {k: float(returns[states==k].mean()) if (states==k).sum()>0 else 0.0
                  for k in range(self.n_regimes)}
        sorted_k = sorted(mean_r, key=lambda k: mean_r[k])  # asc: Bear→Sideways→Bull
        return {sorted_k[i]: ["Bear","Sideways","Bull"][i] for i in range(3)}
