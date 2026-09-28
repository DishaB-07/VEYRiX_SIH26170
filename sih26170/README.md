# AI-Driven Anomaly Detection in Component Burn-In & Screening
### SIH 26170 | ISRO / Department of Space | QA Screening Assistant prototype

> **Synthetic demonstration dataset -- NOT official ISRO data.**
> **Prototype statistical safety criterion -- NOT an official ISRO limit.**
> These two labels appear throughout the app itself; they are repeated here because they matter more than anything else in this README.

---

## A. Project structure

```
sih26170/
├── app.py                      # Streamlit dashboard (entry point)
├── data/                        # (created on first CSV export from the app)
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic burn-in dataset generator + demo-case injector
│   ├── preprocessing.py        # Validation, lot-wise statistics, feature engineering
│   ├── anomaly_detection.py    # Module A: robust Z-score/MAD + Isolation Forest
│   ├── drift_prediction.py     # Module B: 0h+24h -> 168h regression + safety slope
│   ├── evaluation.py           # Lot-wise train/test split + classification/regression metrics
│   ├── explainability.py       # Human-readable explanations + PASS/WATCH/REJECT fusion
│   └── pipeline.py             # Orchestrates the whole flow (no Streamlit dependency -- testable standalone)
├── requirements.txt
└── README.md
```

`pipeline.py` is one small addition beyond the originally sketched structure: it exists purely so the
entire ML pipeline can be imported and smoke-tested from a plain Python script, with no Streamlit
process required. This is what made it possible to catch and fix several real bugs before they ever
reached the UI (see Section G).

## B. What each module does

| File | Responsibility |
|---|---|
| `data_generator.py` | Builds the labelled synthetic burn-in dataset (normal / lot-relative outlier / latent drifter populations, lot-to-lot process variation, measurement noise) and `inject_demo_cases()`, which deterministically overwrites 3 rows of a chosen lot with the CASE A/B/C demo scenarios. |
| `preprocessing.py` | `validate_dataset()` (schema/NaN/non-positive checks, duplicate handling, small-lot warnings), `compute_lot_stats()` (per-lot median/MAD with a population-level fallback for tiny lots), `engineer_features()` (deltas, early slope, lot-relative robust Z per checkpoint, parameter-type dummies). |
| `anomaly_detection.py` | **Module A.** Robust Z-score/MAD per lot per checkpoint (primary, explainable) + Isolation Forest over lot-normalised multi-checkpoint ratios, fit on training lots only (secondary, multivariate backstop). Fuses both into one 0-1 risk score via a transparent weighted-max (not a second model). |
| `drift_prediction.py` | **Module B.** Engineers `Value_0h`, `Value_24h`, deltas, robust-Z, datasheet-limit ratios, and parameter-type dummies; predicts the *relative growth ratio* from 24h to 168h (not the absolute value -- see Section G) via XGBoost (or `HistGradientBoostingRegressor` if XGBoost is unavailable) plus a linear-regression baseline; derives a per-parameter-type safety slope from the training lots' own relative drift-rate distribution (`median + 3*MAD`). |
| `evaluation.py` | `split_by_lot()` (whole lots to train/test, never individual rows); `evaluate_classification()` and `evaluate_regression()`, both guarding degenerate cases (no positives, single class, etc.) instead of throwing. |
| `explainability.py` | Turns Module A/B numbers into the exact style of plain-language QA sentences requested, and fuses both modules' 0-1 scores into a PASS/WATCH/REJECT decision, biased toward recall (a strong single-module flag is never diluted away by the other module looking calm). |
| `pipeline.py` | `run_pipeline()`: validate -> engineer features -> lot-wise split -> fit Module A/B on TRAIN lots only -> score TRAIN and TEST -> evaluate on TEST only. Returns one `PipelineResult` dataclass consumed by the UI. |
| `app.py` | Streamlit UI: Dashboard, Dataset, Lot Analysis, Anomaly Detection, Drift Prediction, QA Decisions (with a pinned Demo Scenario section), Model Performance. |

## C. How to run locally

```bash
cd sih26170
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`). If `xgboost` fails to install
on your machine, delete that line from `requirements.txt` and reinstall -- the app detects its absence
at import time and transparently falls back to `sklearn.ensemble.HistGradientBoostingRegressor`,
labelling whichever model was actually used on the **Drift Prediction** page.

## D. Demo walkthrough (matches the "Load Demo Scenario" button)

1. Click **Load Demo Scenario** in the sidebar. This generates the standard 14-lot synthetic dataset
   (fixed seed, so it's reproducible), determines the lot-wise train/test split, and overwrites 3
   components of the *first held-out test lot* with deterministic CASE A/B/C values -- i.e. these three
   parts are scored by models that have never seen their lot during training.
2. You're taken to **QA Decisions**, where a pinned "Demo scenario" section shows all three, expanded:
   - **CASE A** -- reads well under the datasheet limit at every checkpoint, but is ~2-3x its lot's
     own median from t=0 onward. Module A's robust Z-score flags it; Module B does not (it isn't
     drifting abnormally, it's just already sitting far from its peers) -- **REJECT**.
   - **CASE B** -- an 8-10% early rise by 24h (plausible enough that a human skimming raw numbers could
     wave it through) that becomes a sharp escalation by 168h. Module B's early forecast (built from
     only the 0h/24h readings) catches the projected drift before the 168h data would even exist --
     **REJECT**.
   - **CASE C** -- an unremarkable, normal part. Both modules stay quiet -- **PASS**.
3. Visit **Model Performance** to see the same story in aggregate: recall/FNR on the *held-out* test
   lots, the confusion matrix, and 168h MAE for the gradient-boosting model against a plain linear
   baseline.

## E. Important assumptions

- **No official SIH26170 dataset exists** (the PS's own Dataset Link field is blank). Every row in
  this prototype is synthetically generated and explicitly labelled as such, everywhere in the UI.
- Three parameter types are modelled (`leakage_current_uA`, `iddq_current_uA`,
  `propagation_delay_ns`), each with its own nominal baseline, datasheet limit, and noise level --
  loosely representative of real magnitudes, not sourced from a real qualified part.
- Latent-defect components are constructed to be **statistically close to normal at 0h/24h** and only
  diverge later, by design -- so that catching them early is a genuinely hard, honest test of Module B,
  not a rigged demo.
- The regression target is the **relative (fractional) growth ratio** from 24h to 168h, not the
  absolute 168h value, and the safety slope is derived **per parameter type** in the same relative
  units. This was a deliberate fix after testing showed raw-unit modelling badly overfits to whichever
  parameter type happens to have larger absolute numbers (see Section G).
- Training uses a documented sample-weighting scheme (6x weight on labelled anomalous training rows)
  to reduce false negatives, matching the PS's own emphasis. This is only possible because the
  synthetic data carries labels; a real deployment without ground truth would need a different signal
  (e.g. Module A's own anomaly score) to achieve the same effect.
- Evaluation is always reported on **held-out test lots only**; train-lot metrics are shown separately,
  explicitly labelled as optimistic, purely so the generalisation gap is visible rather than hidden.

## F. Technical limitations (say these out loud before a judge asks)

- **Realism of the synthetic generator is the single biggest open question.** The method is validated
  against the peer-reviewed literature on lot-relative Iddq/leakage screening; the *specific numbers*
  in this generator are illustrative, not measured.
- **Precision is moderate, not high** (typically 30-55% on held-out lots across different random
  splits), while **recall is consistently 100% on every test split we ran**. This is a deliberate,
  documented trade-off matching the PS's own instruction that false negatives are the metric that
  matters most -- but it does mean a real deployment would need a human QA review step for
  WATCH/REJECT items, not a fully automated cutoff.
- **Cold start:** a genuinely brand-new part type with no lot history yet falls back to population-level
  statistics until enough same-type lots accumulate; this is handled gracefully but is not the same as
  having real lot-specific history.
- **No claim of physical failure prediction.** The models forecast a parametric value and a
  statistical risk, not a specific physical failure mechanism or mission-level outcome.
- **The safety slope is a prototype statistic** (`median + 3*MAD` of the training lots' own relative
  drift-rate distribution, per parameter type), not a validated or ISRO-approved reliability limit.

## G. Bugs found and fixed during development (kept here deliberately, for transparency)

Building this honestly surfaced three real issues worth documenting rather than hiding:
1. **A merge collision** in `compute_lot_stats()` when called once per checkpoint (duplicate
   `low_confidence` columns) -- fixed by namespacing the column per checkpoint.
2. **An invalid random range** in the outlier generator when a lot's baseline sits close to its
   parameter's datasheet limit (`rng.uniform(low, high)` with `high < low`) -- fixed by clamping both
   bounds relative to the actual headroom available.
3. **A generalisation bug**: the first working version of Module B predicted the *absolute* 168h value
   pooled across all three parameter types with a single global safety slope in raw units. On held-out
   test lots this made the gradient-boosting model's MAE *worse* than a plain linear baseline, and
   produced wildly inconsistent false-positive rates depending on which parameter type a test lot
   happened to be. Root cause: raw-unit drift rates for `propagation_delay_ns` (~4 units) and
   `iddq_current_uA` (~14 units) are not comparable, and tree models extrapolate poorly to unseen
   absolute baselines. Fix: predict the *relative* growth ratio and derive the safety slope
   *per parameter type*, both in fractional units -- after which the gradient-boosting model
   consistently beat the linear baseline on every held-out split tested, and false-positive rates
   stopped depending on which parameter type was involved.

## H. Tested edge cases

`MAD == 0` (identical readings in a lot), lots smaller than 5 components, lots with only 3 components,
missing/NaN parametric values, duplicate `component_id`s, non-positive readings, a dataset with only
one lot (no valid held-out split -- surfaced as an explicit warning, not a crash), unseen test lots
with zero row overlap with training, and XGBoost being unavailable (verified fallback to
`HistGradientBoostingRegressor`, with the actual model in use displayed on the Drift Prediction page).
A malformed upload missing a required column now fails with a clear, specific message rather than a
raw `KeyError`.

## I. 3-minute SIH presentation flow

| Time | What to show | What to say |
|---|---|---|
| 0:00-0:30 | Dashboard, freshly loaded demo scenario | "Static burn-in screening only asks: did this part cross a fixed line. We built a system that also asks: does this part behave like the rest of its own manufacturing lot -- and where is it headed." |
| 0:30-1:10 | QA Decisions -> CASE A | "This part reads 45-in-a-10 territory, comfortably under the datasheet limit. Static screening waves it through. Our Module A doesn't, because it's using the lot's own statistics, live, not one number written on a datasheet." |
| 1:10-1:50 | QA Decisions -> CASE B | "This one looks almost fine at 24 hours. Our Module B forecasts where it's headed using only the 0h/24h readings, and flags it before the full 168-hour burn-in even finishes -- that's the 'early rejection' the problem statement asks for." |
| 1:50-2:20 | Model Performance | "On lots the model has never seen during training, recall is 100% -- we do not miss a labelled defect in our test data -- and our gradient-boosting forecast beats a plain linear baseline on 168h MAE. Precision is moderate by design: we'd rather over-flag for a human to double-check than let one through." |
| 2:20-2:50 | Anomaly Detection / Drift Prediction pages | "Every flag comes with a plain-language reason -- lot median, robust Z-score, projected drift versus our documented safety criterion -- because a QA inspector needs to be able to act on this, not just trust a black box." |
| 2:50-3:00 | Close | "Everything you've seen runs on a synthetic dataset, clearly labelled as such -- because SIH26170 doesn't provide a real one. The method is what we're presenting; validating it against real burn-in logs is the natural next step." |

## J. 15 likely judge questions, with strong, honest answers

1. **"Is this real ISRO data?"** No -- explicitly, and labelled everywhere in the app. No official
   SIH26170 dataset exists (the PS's Dataset Link is blank); we built a documented synthetic generator
   instead.
2. **"Why should I trust your safety slope?"** It's not a claimed ISRO limit -- it's a transparent
   statistic (`median + 3*MAD` of the training lots' own relative drift-rate distribution, per
   parameter type), displayed as a "prototype statistical criterion" everywhere it appears.
3. **"Why gradient boosting and not an LSTM?"** With only two real input readings per part (0h, 24h),
   there isn't a long sequence for a recurrent model to exploit, and tabular gradient boosting is
   both faster to train and vastly easier to explain via SHAP-style feature attribution -- an explicit
   PS requirement.
4. **"What's your false-negative rate?"** Zero on every held-out test-lot split we ran in development
   -- shown directly via the confusion matrix, not just an aggregate recall percentage.
5. **"Then why isn't precision higher?"** A deliberate, documented trade-off: the PS states false
   negatives are catastrophic, so our fusion logic is biased to surface anything either module flags
   strongly rather than requiring both to agree, which costs precision.
6. **"How do you avoid training on the same lots you test on?"** The split is by whole lot, not by
   row (`split_by_lot`); Module B's regressor, its safety slope, and the Isolation Forest are all fit
   on training lots only, and only ever applied (never re-fit) to test-lot rows.
7. **"What happens with a brand-new part type with no history?"** Falls back to population-level
   median/MAD until enough same-type lots accumulate -- handled explicitly, not silently.
8. **"Could this replace static datasheet limits entirely?"** No, and it shouldn't -- it runs alongside
   them as an additional check, not a replacement; the absolute limit is still enforced.
9. **"Is 125°C / your acceleration reasoning physically grounded?"** It follows the standard Arrhenius
   acceleration-factor model used across electronics reliability engineering; we do not claim a
   specific ISRO-published activation energy.
10. **"What's the single biggest weakness here?"** The synthetic dataset's realism -- every team
    building this PS faces the same gap, and it's the natural next validation step against real
    burn-in logs.
11. **"Why MAD instead of standard deviation?"** MAD is robust to the very outliers it's trying to
    detect; a handful of extreme values barely move the median or MAD, whereas they can inflate a
    classical mean/std enough to mask the anomaly they represent.
12. **"How do you handle a lot that's naturally noisier than others?"** That's exactly what per-lot
    (not global) median/MAD computation is for -- a naturally wide lot gets a wider MAD and therefore
    a fairer threshold, automatically.
13. **"What if a lot only has 2-3 parts?"** Flagged as low-confidence and blended with a
    population-level reference rather than trusted outright -- tested explicitly (see Section H).
14. **"Can a QA inspector understand a rejection without ML background?"** Yes -- every decision
    renders as a specific sentence citing the actual reading, the lot median, the robust Z-score or
    projected drift rate, and the threshold used, not a raw model score.
15. **"What would it take to make this production-ready?"** Real historical burn-in logs to validate
    the generator's assumptions, a human-in-the-loop review workflow for WATCH/REJECT items, and
    engineering sign-off on parameter-specific safety-slope percentiles rather than the single
    prototype multiplier used here.
