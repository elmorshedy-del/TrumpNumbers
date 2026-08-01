"""Turn raw backtest rows into the leaderboard and calibration tables.

The leaderboard lists **every** pre-registered model, including the ones that
lost badly. Poisson's failure is information; hiding it would leave the reader
unable to judge what the winners actually achieved.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import BACKTEST
from ..models.registry import REFERENCE_MODEL, model_catalog
from .scoring import pit_uniformity, skill_score


DEPLOYED_FAMILY = "deployed"

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _cut_labels(cuts) -> dict:
    """Name each cut by the last day it has banked, under the convention."""
    from ..config import CONVENTION

    start = CONVENTION.start_dow
    out = {}
    for c in cuts:
        c = int(c)
        out[str(c)] = (
            "nothing yet" if c < 0 else f"{_DAY_NAMES[(start + c) % 7]} done"
        )
    return out


def _models_only(rows: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered zoo, without the deployed composite scored alongside it.

    `headline-top3` is graded on the same weeks but it is not a candidate: it is
    built *from* the ranking, so letting it into the board it is selected off
    would be circular. It gets its own section.
    """
    if rows.empty or "family" not in rows:
        return rows
    return rows[rows["family"] != DEPLOYED_FAMILY]


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


def skill_by_cut(rows: pd.DataFrame, places: int = 3) -> pd.DataFrame:
    """Skill against climatology *at each cut*, which is the honest comparison.

    Averaged over cuts, a model's CRPS is mostly a statement about the calendar:
    it falls from ~29 on the week's opening day to ~9 on its last, because by
    Saturday six sevenths of the answer has already been handed over. Two models
    can differ on that average purely through which cuts they happen to handle.
    Dividing by the reference model at the same cut removes the calendar and
    leaves the model.
    """
    if rows.empty:
        return pd.DataFrame()
    per = rows.groupby(["model", "cut_dow"])["crps"].mean().unstack("cut_dow")
    if REFERENCE_MODEL not in per.index:
        return pd.DataFrame()
    ref = per.loc[REFERENCE_MODEL]
    return (1.0 - per.div(ref, axis=1)).round(places)


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


def deployed_rule(rows: pd.DataFrame, deployed: str = "headline-top3") -> dict:
    """How the candidate front-page rules score, against the models they use.

    Restricted to the weeks the rules were actually able to run — they need a
    ranking, so they stand down until there are enough closed weeks to build
    one. Comparing their average against models scored over a longer span would
    be comparing two different samples of weeks.
    """
    dep = rows[rows["family"] == DEPLOYED_FAMILY] if "family" in rows else rows.iloc[0:0]
    if dep.empty:
        return {}
    weeks = set(dep["week"].unique())
    same = rows[rows["week"].isin(weeks)]
    # The composites must not appear in the board they are compared against.
    board = leaderboard(_models_only(same))
    if board.empty:
        return {}

    agg = dep.groupby("model").agg(
        crps=("crps", "mean"),
        coverage80=("covered80", "mean"),
        width80=("width80", "mean"),
        n=("crps", "size"),
    ).sort_values("crps")
    best_name = board.index[0]
    ref = board.loc[REFERENCE_MODEL, "crps"] if REFERENCE_MODEL in board.index else float("nan")

    candidates = [
        {
            "rule": name,
            "crps": round(float(r["crps"]), 2),
            "coverage80": round(float(r["coverage80"]), 3),
            "width80": round(float(r["width80"]), 1),
            "crps_skill_vs_climatology": round(float(skill_score(r["crps"], ref)), 3),
            "deployed": bool(name == deployed),
        }
        for name, r in agg.iterrows()
    ]
    current = next((c for c in candidates if c["deployed"]), candidates[0])
    return {
        "deployed": deployed,
        "n_forecasts": int(dep[dep["model"] == current["rule"]].shape[0]),
        "n_weeks": int(dep["week"].nunique()),
        "weeks_from": min(weeks).date().isoformat() if weeks else None,
        "candidates": candidates,
        "best_rule": candidates[0]["rule"],
        "crps": current["crps"],
        "coverage80": current["coverage80"],
        "width80": current["width80"],
        "crps_skill_vs_climatology": current["crps_skill_vs_climatology"],
        "best_single_model_on_same_weeks": best_name,
        "best_single_model_crps": round(float(board.loc[best_name, "crps"]), 2),
        "note": (
            "The rule the front page deploys, and the alternatives it was never measured "
            "against, each replayed week by week with the ranking recomputed from earlier "
            "weeks only — so no rule selects using the week it is graded on. Compared here "
            "against every model over exactly the same weeks. A combination is not entitled "
            "to its members' scores."
        ),
    }


def build_report(result) -> dict:
    rows = result.rows
    if rows.empty:
        return {"error": "no backtest rows"}

    dev, hold = holdout_split(_models_only(rows), result.holdout_start)
    # `leaderboard` is what `pipeline.load_ranking` reads to pick the live
    # models, so it has to be the DEVELOPMENT board and nothing else. It was
    # being built over every row — holdout weeks included — while the notes
    # beside it said the holdout never selects. The eight holdout weeks were
    # one seventh of the selection sample, which is a smaller version of
    # exactly the leak this file already claims to have closed.
    lb = leaderboard(dev if not dev.empty else _models_only(rows))
    lb_all = leaderboard(_models_only(rows))
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

    cuts = by_cut(_models_only(rows))
    cover_cuts = by_cut(_models_only(rows), "covered80", places=3)
    skill_cuts = skill_by_cut(_models_only(rows))

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
        # Development weeks only. This is the board the live headline selects
        # from, so it must not contain the holdout.
        "leaderboard": pack(lb),
        "leaderboard_holdout": pack(leaderboard(hold)) if not hold.empty else [],
        # Every scored week, for reading only — never for selecting.
        "leaderboard_all_weeks": pack(lb_all),
        "deployed_rule": deployed_rule(rows, deployed=f"headline-top{BACKTEST.headline_top_k}"),
        "n_weeks": int(rows["week"].nunique()),
        "n_forecasts": int(len(rows)),
        "n_weeks_development": int(dev["week"].nunique()) if not dev.empty else 0,
        "holdout_start": (
            result.holdout_start.date().isoformat() if result.holdout_start is not None else None
        ),
        "thresholds": [round(t, 1) for t in result.thresholds],
        "crps_by_cut": spread(cuts),
        "coverage80_by_cut": spread(cover_cuts),
        "skill_by_cut": spread(skill_cuts),
        "cut_days": [int(c) for c in sorted(rows["cut_dow"].unique())],
        # A cut is a POSITION in the week, not a pandas weekday, and the site
        # was labelling the columns Mon..Sun on both counts wrongly: it started
        # at Monday under a Sunday-anchored convention and had no column for the
        # -1 cut at all, so the hardest forecast on the board was invisible and
        # every other column was named after the wrong day.
        "cut_labels": _cut_labels(sorted(int(c) for c in rows["cut_dow"].unique())),
        "calibration": [
            calibration_detail(rows, m) for m in lb.index[:6]
        ],
        "reference_model": REFERENCE_MODEL,
        "notes": [
            "Every pre-registered model is listed, including those that lost badly.",
            (
                "`leaderboard` covers the development weeks only — it is what the live headline "
                "selects from, so the holdout is kept out of it. `leaderboard_all_weeks` is the "
                "same board over every scored week and is for reading, not selecting."
            ),
            (
                "`deployed_rule` scores the composite the front page actually shows, week by "
                "week, selecting only from weeks that had already closed. Its members' scores "
                "are not its score."
            ),
            (
                "`skill_by_cut` divides each model's CRPS by climatology's at the SAME cut. The "
                "overall average is dominated by how much of the week was already known, which "
                "is a fact about the calendar rather than about the model."
            ),
            (
                "Draws are seeded from (model, week, cut), so this board reproduces exactly on "
                "unchanged data. Re-running under a different seed is how to measure the "
                "Monte-Carlo spread rather than mistake it for a ranking."
            ),
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
