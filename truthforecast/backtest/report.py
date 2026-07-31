"""Turn raw backtest rows into the leaderboard and calibration tables.

The leaderboard lists **every** pre-registered model, including the ones that
lost badly. Poisson's failure is information; hiding it would leave the reader
unable to judge what the winners actually achieved.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..models.registry import REFERENCE_MODEL, model_catalog
from .scoring import pit_uniformity, skill_score


def leaderboard(rows: pd.DataFrame, cut_dow: int | None = None) -> pd.DataFrame:
    """Mean scores per model, optionally for a single weekday cut."""
    df = rows if cut_dow is None else rows[rows["cut_dow"] == cut_dow]
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("model").agg(
        family=("family", "first"),
        n=("crps", "size"),
        crps=("crps", "mean"),
        log_score=("log_score", "mean"),
        pinball=("pinball", "mean"),
        width80=("width80", "mean"),
        coverage80=("covered80", "mean"),
        coverage95=("covered95", "mean"),
        median_abs_err=("median", lambda s: np.mean(np.abs(s.to_numpy() - df.loc[s.index, "actual"].to_numpy()))),
    )

    ref = agg.loc[REFERENCE_MODEL, "crps"] if REFERENCE_MODEL in agg.index else np.nan
    agg["crps_skill_vs_climatology"] = [skill_score(c, ref) for c in agg["crps"]]

    # Calibration: is the 80% interval actually an 80% interval?
    agg["calibration_error_80"] = (agg["coverage80"] - 0.80).abs()

    pit = df.groupby("model")["pit"].apply(lambda s: pit_uniformity(s.to_numpy()))
    agg["pit_p_value"] = [pit[m].get("p_value") for m in agg.index]
    agg["pit_calibrated"] = [pit[m].get("calibrated") for m in agg.index]

    return agg.sort_values("crps").round(4)


def by_cut(rows: pd.DataFrame, column: str = "crps", places: int = 2) -> pd.DataFrame:
    """How does a metric move as the week reveals itself?

    Reported per cut rather than only as an average, because the average over
    cuts is dominated by how much of the week is already known and can hide a
    model that is fine on Saturday and badly overconfident on Monday.
    """
    if rows.empty or column not in rows:
        return pd.DataFrame()
    return (
        rows.groupby(["model", "cut_dow"])[column].mean().unstack("cut_dow").round(places)
    )


def calibration_detail(rows: pd.DataFrame, model: str) -> dict:
    sub = rows[rows["model"] == model]
    if sub.empty:
        return {}
    return {
        "model": model,
        "pit": pit_uniformity(sub["pit"].to_numpy()),
        "coverage": {
            "80": round(float(sub["covered80"].mean()), 3),
            "95": round(float(sub["covered95"].mean()), 3),
        },
        "mean_width80": round(float(sub["width80"].mean()), 1),
    }


def holdout_split(rows: pd.DataFrame, holdout_start) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development weeks vs the untouched final segment.

    The holdout is the honest number. Everything else has been looked at while
    the models were being built, and looking is itself a form of fitting.
    """
    if holdout_start is None:
        return rows, rows.iloc[0:0]
    return rows[rows["week"] < holdout_start], rows[rows["week"] >= holdout_start]


def build_report(result) -> dict:
    rows = result.rows
    if rows.empty:
        return {"error": "no backtest rows"}

    dev, hold = holdout_split(rows, result.holdout_start)
    lb = leaderboard(rows)
    catalog = {m["name"]: m for m in model_catalog()}

    def pack(frame: pd.DataFrame) -> list[dict]:
        if frame.empty:
            return []
        out = []
        for name, r in frame.iterrows():
            meta = catalog.get(name, {})
            out.append(
                {
                    "model": name,
                    "family": r["family"],
                    "n_forecasts": int(r["n"]),
                    "crps": round(float(r["crps"]), 2),
                    "crps_skill_vs_climatology": round(float(r["crps_skill_vs_climatology"]), 3),
                    "log_score": round(float(r["log_score"]), 3),
                    "pinball": round(float(r["pinball"]), 3),
                    "width80": round(float(r["width80"]), 1),
                    "coverage80": round(float(r["coverage80"]), 3),
                    "coverage95": round(float(r["coverage95"]), 3),
                    "pit_p_value": (
                        None if r["pit_p_value"] is None or not np.isfinite(r["pit_p_value"])
                        else round(float(r["pit_p_value"]), 4)
                    ),
                    "pit_calibrated": bool(r["pit_calibrated"]) if r["pit_calibrated"] is not None else None,
                    "description": meta.get("description", ""),
                    "assumptions": meta.get("assumptions", ""),
                    "fails_when": meta.get("fails_when", ""),
                }
            )
        return out

    cuts = by_cut(rows)
    cover_cuts = by_cut(rows, "covered80", places=3)

    def spread(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {}
        return {
            m: {str(c): (None if pd.isna(v) else float(v)) for c, v in row.items()}
            for m, row in frame.iterrows()
        }

    return {
        # A leaderboard has no expiry stamped on it, so it can outlive the data
        # it describes without looking any different. This is that stamp.
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard": pack(lb),
        "leaderboard_holdout": pack(leaderboard(hold)) if not hold.empty else [],
        "n_weeks": int(rows["week"].nunique()),
        "n_forecasts": int(len(rows)),
        "holdout_start": (
            result.holdout_start.date().isoformat() if result.holdout_start is not None else None
        ),
        "thresholds": [round(t, 1) for t in result.thresholds],
        "crps_by_cut": spread(cuts),
        "coverage80_by_cut": spread(cover_cuts),
        "cut_days": [int(c) for c in sorted(rows["cut_dow"].unique())],
        "calibration": [
            calibration_detail(rows, m) for m in lb.index[:6]
        ],
        "reference_model": REFERENCE_MODEL,
        "notes": [
            "Every pre-registered model is listed, including those that lost badly.",
            "CRPS is the primary metric; lower is better and it is in units of posts.",
            "Skill is measured against climatology (empirical-dow): 0.10 means 10% of its error removed.",
            "coverage80 should be ~0.80. Much below means overconfident intervals.",
            "PIT p-value below 0.05 means the forecast distribution is NOT calibrated.",
            "The holdout leaderboard is the honest number; the main one was visible during development.",
            (
                "The live headline picks its models from the development leaderboard, never from "
                "the holdout. Selecting on the holdout would spend the one unlooked-at sample on "
                "a choice and then report the result as if it were still out-of-sample."
            ),
            (
                "Cuts run from -1 (Monday, nothing of the week observed) to 5 (Saturday complete) "
                "— exactly the situations the site is ever in. The fully-observed Sunday cut is "
                "excluded because every model returns the answer there: it scores 0 CRPS, covers "
                "its own interval and reports zero width, which would flatter coverage and shrink "
                "CRPS for every model alike."
            ),
            (
                "Each week is scored at seven weekday cuts, so the forecasts are NOT independent. "
                "Treat the PIT p-values as a strong directional signal about over/under-confidence, "
                "not as an exact significance level — the effective sample size is closer to the "
                "week count than to the forecast count."
            ),
            (
                "Differences of a few hundredths of a CRPS point between neighbouring models are "
                "well inside the noise of ~50 scored weeks. Read the groupings, not the ordering."
            ),
        ],
    }
