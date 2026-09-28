"""
Synthetic burn-in / ESS dataset generator for SIH26170.

IMPORTANT: SIH26170 does not provide an official public dataset. Every array
produced here is SYNTHETIC and clearly labelled as such throughout the app.
It is designed to demonstrate the *method*, not to represent real ISRO or
component measurements.

Design goals (see accompanying research notes):
  - Normal parts drift a little (real burn-in isn't perfectly flat).
  - "Lot-relative outliers" are elevated from 0h onward but still usually
    stay under the absolute datasheet limit -> only lot-relative statistics
    (Module A) can catch them.
  - "Latent drifters" look close to normal at 0h/24h and only diverge later
    via an accelerating (super-linear) drift term -> this is what Module B's
    early-reading forecast is meant to catch before 168h.
  - Lot-to-lot process variation is built in (each lot has its own baseline
    median and spread), so a single global threshold would not work well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATASET_LABEL = "Synthetic demonstration dataset \u2014 NOT official ISRO data."

# Nominal per-parameter settings: (baseline_median, datasheet_limit, rel_noise)
PARAMETER_PROFILES = {
    "leakage_current_uA": dict(baseline=10.0, limit=50.0, rel_noise=0.06),
    "iddq_current_uA": dict(baseline=14.0, limit=80.0, rel_noise=0.07),
    "propagation_delay_ns": dict(baseline=4.0, limit=12.0, rel_noise=0.05),
}

COMPONENT_TYPES = ["ASIC-R7", "FPGA-K3", "OPAMP-L2", "MOSFET-D9"]

CHECKPOINTS_H = [0, 24, 96, 168]


def _lot_baseline(rng: np.random.Generator, profile: dict) -> tuple[float, float]:
    """Each lot gets its own baseline median and its own process spread
    (lot-to-lot variation) so that a single global threshold is never
    appropriate -- this is the whole premise of Module A."""
    median = profile["baseline"] * rng.lognormal(mean=0.0, sigma=0.18)
    spread_cv = rng.uniform(0.05, 0.14)  # coefficient of variation within lot
    return float(median), float(spread_cv)


def _normal_trajectory(rng, v0, hours, rel_noise):
    """Mild, mostly-linear upward drift with multiplicative noise at every
    checkpoint -- real burn-in parts do drift a little, not zero."""
    drift_rate_per_h = rng.uniform(0.00005, 0.00025)  # ~fractional / hour
    vals = []
    for h in hours:
        trend = v0 * (1 + drift_rate_per_h * h)
        noisy = trend * (1 + rng.normal(0, rel_noise))
        vals.append(max(noisy, 1e-6))
    return vals


def _lot_relative_outlier_trajectory(rng, lot_median, hours, rel_noise, limit):
    """Elevated from t=0 onward (process/material outlier), stays roughly
    flat-elevated, and is constructed to usually remain UNDER the absolute
    datasheet limit -- this is the exact 45uA-in-a-10uA-lot scenario."""
    # Keep the multiplier bounded so v0 typically stays under the limit.
    max_mult = max(1.5, (limit * 0.94) / max(lot_median, 1e-6))
    high_mult = min(5.5, max_mult)
    low_mult = min(2.8, high_mult * 0.85)
    if high_mult <= low_mult:
        high_mult = low_mult + 0.1
    mult = rng.uniform(low_mult, high_mult)
    v0 = lot_median * mult
    drift_rate_per_h = rng.uniform(0.00005, 0.00030)
    vals = []
    for h in hours:
        trend = v0 * (1 + drift_rate_per_h * h)
        noisy = trend * (1 + rng.normal(0, rel_noise))
        vals.append(max(noisy, 1e-6))
    return vals


def _latent_drifter_trajectory(rng, v0_normal_like, hours, rel_noise, limit):
    """Near-normal at 0h/24h (overlaps with the normal population -- this is
    what makes it "latent"), then an accelerating drift term that only
    becomes visually obvious by 96h/168h. Roughly half will still land under
    the absolute limit at 168h (the true "escapes static screening" case);
    the rest cross it (still useful for calibrating recall)."""
    v0 = v0_normal_like * (1 + rng.normal(0, rel_noise * 0.6))
    k = rng.uniform(0.9, 2.6)   # drift magnitude
    p = rng.uniform(1.6, 2.6)   # super-linearity exponent
    vals = []
    for h in hours:
        accel = k * (h / 168.0) ** p
        trend = v0 * (1 + accel)
        noisy = trend * (1 + rng.normal(0, rel_noise))
        vals.append(max(noisy, 1e-6))
    return vals


def generate_synthetic_dataset(
    n_lots: int = 14,
    min_parts_per_lot: int = 18,
    max_parts_per_lot: int = 40,
    outlier_frac: float = 0.06,
    drifter_frac: float = 0.06,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the full synthetic burn-in dataset. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    rows = []
    param_names = list(PARAMETER_PROFILES.keys())

    for lot_idx in range(1, n_lots + 1):
        lot_id = f"LOT_{lot_idx:02d}"
        param_name = param_names[(lot_idx - 1) % len(param_names)]
        profile = PARAMETER_PROFILES[param_name]
        component_type = rng.choice(COMPONENT_TYPES)
        lot_median, spread_cv = _lot_baseline(rng, profile)
        n_parts = int(rng.integers(min_parts_per_lot, max_parts_per_lot + 1))

        # Decide per-part category
        n_outliers = max(0, int(round(n_parts * rng.uniform(0.0, outlier_frac * 2))))
        n_drifters = max(0, int(round(n_parts * rng.uniform(0.0, drifter_frac * 2))))
        n_outliers = min(n_outliers, n_parts // 4)
        n_drifters = min(n_drifters, n_parts // 4)
        n_normal = n_parts - n_outliers - n_drifters

        categories = (
            ["normal"] * n_normal
            + ["lot_relative_outlier"] * n_outliers
            + ["latent_drifter"] * n_drifters
        )
        rng.shuffle(categories)

        for part_idx, category in enumerate(categories, start=1):
            component_id = f"{lot_id}-C{part_idx:03d}"
            v0_normal_like = lot_median * (1 + rng.normal(0, spread_cv))
            v0_normal_like = max(v0_normal_like, 1e-6)

            if category == "normal":
                vals = _normal_trajectory(rng, v0_normal_like, CHECKPOINTS_H, profile["rel_noise"])
            elif category == "lot_relative_outlier":
                vals = _lot_relative_outlier_trajectory(
                    rng, lot_median, CHECKPOINTS_H, profile["rel_noise"], profile["limit"]
                )
            else:  # latent_drifter
                vals = _latent_drifter_trajectory(
                    rng, v0_normal_like, CHECKPOINTS_H, profile["rel_noise"], profile["limit"]
                )

            row = {
                "component_id": component_id,
                "lot_id": lot_id,
                "component_type": component_type,
                "parameter_type": param_name,
                "Value_0h": round(vals[0], 4),
                "Value_24h": round(vals[1], 4),
                "Value_96h": round(vals[2], 4),
                "Value_168h": round(vals[3], 4),
                "datasheet_limit": profile["limit"],
                "ground_truth_anomaly": category != "normal",
                "anomaly_type": category,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


def inject_demo_cases(df: pd.DataFrame, target_lot_id: str, seed: int = 7) -> pd.DataFrame:
    """Overwrite three existing rows inside `target_lot_id` with hand-crafted,
    DETERMINISTIC components that reproduce the three headline demo
    scenarios. Replacing rows inside an existing, realistically-sized lot
    (rather than inventing a brand-new 3-part lot) keeps the lot's own
    median/MAD statistics stable and meaningful. Deterministic multipliers
    (rather than random draws) are used so the demo reliably reproduces the
    same PASS/WATCH/REJECT outcome every run, for any lot/parameter type --
    verified empirically across all synthetic lots during development.

    CASE A: passes the absolute datasheet limit, but is a clear lot-relative
            outlier from t=0 (Module A should catch it).
    CASE B: modestly, plausibly elevated at 0h/24h (an 8-10% early rise --
            not zero-signal, but well within what a QA inspector could
            plausibly wave through) and then escalates sharply by 168h
            (Module B's early forecast should catch it before 168h data
            even exists).
    CASE C: a clean, normal part that should pass both modules.
    """
    df = df.copy()
    lot_rows = df[df["lot_id"] == target_lot_id]
    if len(lot_rows) < 6:
        raise ValueError("target_lot_id must reference a lot with >= 6 components")

    profile_name = lot_rows["parameter_type"].iloc[0]
    profile = PARAMETER_PROFILES[profile_name]
    lot_median = float(lot_rows["Value_0h"].median())
    limit = profile["limit"]
    idxs = lot_rows.index[:3]

    # CASE A -- lot-relative outlier, safely under the absolute limit,
    # elevated from t=0 and staying roughly flat-elevated thereafter.
    high_mult = min(4.0, (limit * 0.85) / max(lot_median, 1e-6))
    a_mult = max(1.6, high_mult)
    a_schedule = [1.0, 1.02, 1.05, 1.07]  # mild additional drift on top of a_mult
    a_vals = [lot_median * a_mult * s for s in a_schedule]
    df.loc[idxs[0], ["Value_0h", "Value_24h", "Value_96h", "Value_168h"]] = [round(v, 4) for v in a_vals]
    df.loc[idxs[0], "component_id"] = "DEMO-CASE-A"
    df.loc[idxs[0], "anomaly_type"] = "lot_relative_outlier"
    df.loc[idxs[0], "ground_truth_anomaly"] = True

    # CASE B -- latent drifter: this exact multiplier schedule
    # (1.0 / 1.10 / 1.32 / 1.70 x the lot median) was empirically verified
    # to trip Module B's early forecast across every synthetic lot/parameter
    # type generated by this module.
    b_schedule = [1.0, 1.10, 1.32, 1.70]
    b_vals = [lot_median * s for s in b_schedule]
    df.loc[idxs[1], ["Value_0h", "Value_24h", "Value_96h", "Value_168h"]] = [round(v, 4) for v in b_vals]
    df.loc[idxs[1], "component_id"] = "DEMO-CASE-B"
    df.loc[idxs[1], "anomaly_type"] = "latent_drifter"
    df.loc[idxs[1], "ground_truth_anomaly"] = True

    # CASE C -- clean, unremarkable normal part
    c_schedule = [1.0, 1.01, 1.02, 1.03]
    c_vals = [lot_median * s for s in c_schedule]
    df.loc[idxs[2], ["Value_0h", "Value_24h", "Value_96h", "Value_168h"]] = [round(v, 4) for v in c_vals]
    df.loc[idxs[2], "component_id"] = "DEMO-CASE-C"
    df.loc[idxs[2], "anomaly_type"] = "normal"
    df.loc[idxs[2], "ground_truth_anomaly"] = False

    return df
