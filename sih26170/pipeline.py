"""SIH26170 - burn-in anomaly screening. Synthetic data + Module A + Module B + explanations."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

PARAMS = {  # name: (unit, typical lot baseline)  -- SYNTHETIC, not real ISRO values
    "Iddq": ("uA", 8.0), "Leakage": ("uA", 10.0), "PropDelay": ("ns", 12.0)}
T = [0, 24, 96, 168]
MAD_K = 1.4826

# ---------------------------------------------------------------- data
def generate(n_lots=14, parts=40, defect_rate=0.08, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for l in range(n_lots):
        p = list(PARAMS)[l % 3]; unit, base = PARAMS[p]
        lot_base = base * rng.lognormal(0, 0.15)          # lot-to-lot shift
        limit = lot_base * 5
        for i in range(parts):
            bad = rng.random() < defect_rate
            v0 = lot_base * rng.lognormal(0, 0.06)
            if bad and rng.random() < 0.5:                 # some defects already elevated at 0h
                v0 = min(v0 * rng.uniform(2.0, 3.5), limit * 0.9)
            a = rng.uniform(0.01, 0.05)                    # normal mild upward drift
            d = rng.uniform(0.4, 1.6) if bad else 0.0
            pw = rng.uniform(1.3, 2.0)
            vals = []
            for t in T:
                x = t / 168
                v = v0 * (1 + a * x + d * x ** pw) * rng.lognormal(0, 0.008)
                vals.append(min(v, limit * 0.98) if bad else v)  # keep latent defects inside limit
            rows.append(dict(component_id=f"L{l:02d}-{i:03d}", lot_id=f"L{l:02d}", parameter=p,
                             unit=unit, datasheet_limit=limit, v0=vals[0], v24=vals[1],
                             v96=vals[2], v168=vals[3], is_defect=int(bad)))
    return pd.DataFrame(rows)

REQUIRED = ["component_id", "lot_id", "parameter", "datasheet_limit", "v0", "v24"]
def validate(df):
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {miss}")
    df = df.dropna(subset=["v0", "v24"]).copy()
    return df[(df.v0 > 0) & (df.v24 > 0)].reset_index(drop=True)

# ---------------------------------------------------------------- Module A
def module_a(df, z_watch=3.5, z_reject=6.0):
    df = df.copy()
    for c in ("v0", "v24"):
        lg = np.log(df[c]); g = lg.groupby([df.lot_id, df.parameter])
        med = g.transform("median")
        mad = (lg - med).abs().groupby([df.lot_id, df.parameter]).transform("median") * MAD_K
        df[f"z_{c}"] = (lg - med) / mad.clip(lower=0.02)          # floor avoids MAD=0 blowups
        df[f"x_{c}"] = np.exp(med)                                  # lot median (for explanation)
    df["robust_z"] = df[["z_v0", "z_v24"]].max(axis=1)
    df["dz"] = df.z_v24 - df.z_v0
    df["iso"] = 0.0
    for p, g in df.groupby("parameter"):                            # multivariate backstop
        if len(g) < 20: continue
        m = IsolationForest(n_estimators=200, contamination=0.05, random_state=0)
        m.fit(g[["z_v0", "z_v24", "dz"]])
        s = -m.score_samples(g[["z_v0", "z_v24", "dz"]])
        df.loc[g.index, "iso"] = (s - s.min()) / (np.ptp(s) + 1e-9)
    df["a_flag"] = np.select([df.robust_z >= z_reject, df.robust_z >= z_watch], [2, 1], 0)
    return df

# ---------------------------------------------------------------- Module B
def _feats(df):
    lg0, lg24 = np.log(df.v0), np.log(df.v24)
    return np.c_[lg0, lg24, lg24 - lg0, df.get("z_v0", 0), df.get("z_v24", 0)]

class DriftModel:
    """Predicts log growth factor log(v168/v24); safety slope = percentile of predicted drift rate."""
    def __init__(self, pct=90): self.pct = pct
    def fit(self, df):
        y = np.log(df.v168 / df.v24); X = _feats(df)
        self.gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                             subsample=0.8, random_state=0).fit(X, y)
        self.lin = LinearRegression().fit(X[:, :3], y)
        rate = self.gbm.predict(X) / 144
        self.slope = {p: np.percentile(rate[(df.parameter == p).values], self.pct) for p in df.parameter.unique()}
        return self
    def predict(self, df):
        df = df.copy(); X = _feats(df)
        df["pred168"] = df.v24 * np.exp(self.gbm.predict(X))
        df["pred168_lin"] = df.v24 * np.exp(self.lin.predict(X[:, :3]))
        df["drift_rate"] = self.gbm.predict(X) / 144           # relative growth per hour
        df["safety_slope"] = df.parameter.map(self.slope).fillna(np.median(list(self.slope.values())))
        df["b_flag"] = (df.drift_rate > df.safety_slope).astype(int)
        return df

# ---------------------------------------------------------------- fusion + explanation
def decide(df):
    df = df.copy()
    df["risk"] = np.clip(0.6 * np.clip(df.robust_z / 6, 0, 1) + 0.25 * df.b_flag + 0.15 * df.iso, 0, 1)
    strong = (df.a_flag == 2)
    df["decision"] = np.select([strong | ((df.a_flag == 1) & (df.b_flag == 1)),
                                (df.a_flag == 1) | (df.b_flag == 1) | (df.iso > 0.75)],
                               ["REJECT", "WATCH"], "NORMAL")
    df["reason"] = df.apply(explain, axis=1)
    return df

def explain(r):
    out = []
    if r.a_flag:
        w = "v0" if r.z_v0 >= r.z_v24 else "v24"; h = "0h" if w == "v0" else "24h"
        ratio = r[w] / r[f"x_{w}"]
        out.append(f"{r.parameter} at {h} is {ratio:.1f}x the lot median (robust Z = {r.robust_z:.1f} sigma)")
        if r[w] < r.datasheet_limit: out.append(f"still under datasheet limit ({r[w]:.1f} < {r.datasheet_limit:.1f} {r.unit})")
    if r.b_flag:
        out.append(f"predicted 168h = {r.pred168:.1f} {r.unit}; drift rate is {r.drift_rate / r.safety_slope:.1f}x the {r.parameter} safety slope")
    if r.iso > 0.75 and not out: out.append("multivariate pattern unusual vs lot (Isolation Forest)")
    return "; ".join(out) or "consistent with lot peers"

def run(df, model=None):
    df = module_a(validate(df))
    if model is None: model = DriftModel().fit(df.dropna(subset=["v168"]))
    return decide(model.predict(df)), model

# ---------------------------------------------------------------- evaluation
def evaluate(df):
    """Train Module B on 70% of lots, evaluate on held-out lots."""
    lots = sorted(df.lot_id.unique()); rng = np.random.default_rng(1); rng.shuffle(lots)
    test_lots = lots[: max(1, int(len(lots) * 0.3))]
    a = module_a(df); tr, te = a[~a.lot_id.isin(test_lots)], a[a.lot_id.isin(test_lots)]
    model = DriftModel().fit(tr); out = decide(model.predict(te))
    y, p = out.is_defect, (out.decision != "NORMAL").astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return dict(recall=tp / max(tp + fn, 1), fnr=fn / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                tp=tp, fn=fn, fp=fp, tn=tn, test_lots=test_lots,
                mae_gbm=mean_absolute_error(out.v168, out.pred168),
                mae_lin=mean_absolute_error(out.v168, out.pred168_lin),
                mae_naive=mean_absolute_error(out.v168, out.v24)), out
