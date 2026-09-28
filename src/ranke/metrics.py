import math

import pandas as pd

from .cases import DIMENSIONS, SPLITS

MIN_PER_LABEL = 30


def grade(score, fail_below, pass_from):
    if score < fail_below:
        return "fail"
    if score >= pass_from:
        return "pass"
    return "review"


def _rate(part, whole):
    return part / whole if whole else float("nan")


def agreement(labels, grades):
    labels = list(labels)
    grades = list(grades)
    fails = [g for l, g in zip(labels, grades) if l == "fail"]
    passes = [g for l, g in zip(labels, grades) if l == "pass"]
    return {
        "expert fails": len(fails),
        "expert passes": len(passes),
        "catch rate": _rate(fails.count("fail"), len(fails)),
        "fails sent to review": _rate(fails.count("review"), len(fails)),
        "fails missed": _rate(fails.count("pass"), len(fails)),
        "correct-pass rate": _rate(passes.count("pass"), len(passes)),
        "passes sent to review": _rate(passes.count("review"), len(passes)),
        "false alarms": _rate(passes.count("fail"), len(passes)),
    }


def balanced(stats):
    values = [stats["catch rate"], stats["correct-pass rate"]]
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def test_report(cases, grade_col="Ranke grade"):
    labeled = cases[cases["Expert label"].isin(["pass", "fail"]) & (cases[grade_col] != "skipped")]
    by_split = {}
    for split in SPLITS + ("all",):
        part = labeled if split == "all" else labeled[labeled["Split"] == split]
        if len(part):
            by_split[split] = agreement(part["Expert label"], part[grade_col])
    warnings = []
    for split, stats in by_split.items():
        if split in ("tuning", "all"):
            continue
        for key, name in (("expert fails", "fail"), ("expert passes", "pass")):
            if stats[key] < MIN_PER_LABEL:
                warnings.append(
                    f"The {split} split has only {stats[key]} expert {name} case(s). "
                    f"Aim for at least {MIN_PER_LABEL} before trusting its numbers."
                )
    dims = {}
    for dim in DIMENSIONS:
        values = labeled[labeled[dim] != ""]
        if values.empty:
            continue
        rows = []
        for value, group in values.groupby(dim):
            stats = agreement(group["Expert label"], group[grade_col])
            rows.append({dim: value, **stats})
        dims[dim] = pd.DataFrame(rows)
    return {"by_split": by_split, "dimensions": dims, "warnings": warnings, "rows": len(labeled)}


def weak_spots(dimensions, threshold=0.7, min_fails=3):
    spots = []
    for dim, table in dimensions.items():
        for _, row in table.iterrows():
            if row["expert fails"] >= min_fails and row["catch rate"] < threshold:
                spots.append(f"{dim} = {row[dim]}: catches only {row['catch rate']:.0%} of made-up cases")
    return spots


def eval_report(cases, grade_col="Ranke grade"):
    graded = cases[cases[grade_col].isin(["pass", "fail", "review"])]

    def summary(part):
        n = len(part)
        return {
            "rows": n,
            "made-up rate": _rate((part[grade_col] == "fail").sum(), n),
            "needs review": _rate((part[grade_col] == "review").sum(), n),
            "looks supported": _rate((part[grade_col] == "pass").sum(), n),
        }

    dims = {}
    for dim in DIMENSIONS:
        values = graded[graded[dim] != ""]
        if values.empty:
            continue
        dims[dim] = pd.DataFrame([{dim: v, **summary(g)} for v, g in values.groupby(dim)])
    return {"overall": summary(graded), "dimensions": dims, "skipped": int((cases[grade_col] == "skipped").sum())}


def correlation(a, b):
    a, b = list(a), list(b)
    if len(a) < 3:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b)))
