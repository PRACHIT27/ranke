import math
from datetime import date

import pandas as pd

from . import __version__
from .cases import assign_splits
from .grader import DEFAULT_OPPOSITE, DEFAULT_QUESTION, AutoGrader, grade_row
from .metrics import agreement, balanced, correlation, eval_report, grade, test_report, weak_spots

FLIP_SAMPLE = 30
FLIP_WARN = 0.5


def questions_from(run_info):
    asked = [run_info.get(k, "").strip() for k in ("Laya question", "Laya question 2", "Laya question 3")]
    asked = [q for q in asked if q]
    if not asked:
        return [DEFAULT_QUESTION], DEFAULT_OPPOSITE
    opposite = run_info.get("Laya question (opposite)", "").strip()
    if not opposite and asked[0] == DEFAULT_QUESTION:
        opposite = DEFAULT_OPPOSITE
    return asked, opposite or None


def _usable(row):
    if not row["Context"]:
        return "Skipped: Context is empty."
    if not row["AI output"]:
        return "Skipped: AI output is empty."
    return ""


def _notes(scored, show_grader=False):
    notes = [f"Graded by {scored['grader']}."] if show_grader else []
    if scored["chunks"] > 1:
        notes.append(f"Context split into {scored['chunks']} pieces.")
    if scored["sentences"] > 1:
        notes.append(f"Output checked as {scored['sentences']} sentences.")
    if scored["cut_off"]:
        notes.append("Warning: a piece was longer than Laya can read and was cut off.")
    return " ".join(notes)


def _score_rows(grader, cases, mask, questions, progress, with_input):
    scores = {i: [None] * len(questions) for i in cases.index[mask]}
    weakest = {i: [""] * len(questions) for i in cases.index[mask]}
    notes = {}
    todo = list(cases.index[mask])
    for n, i in enumerate(todo, 1):
        row = cases.loc[i]
        reason = _usable(row)
        if reason:
            notes[i] = reason
            continue
        request = row["User input"] if with_input else ""
        scored = grade_row(grader, row["Context"], row["AI output"], request, questions)
        for q, (value, sentence) in enumerate(scored["results"]):
            scores[i][q] = value
            weakest[i][q] = sentence
        notes[i] = _notes(scored, isinstance(grader, AutoGrader))
        if progress:
            progress(n, len(todo))
    return scores, weakest, notes


def _apply(cases, mask, scores, weakest, notes, q, fail_below, pass_from):
    for col in ("Ranke score", "Ranke grade", "Weakest sentence", "Matches expert", "Ranke note"):
        if col not in cases.columns:
            cases[col] = ""
        cases[col] = cases[col].astype(object)
    for i in cases.index[mask]:
        value = scores[i][q]
        cases.at[i, "Ranke note"] = notes.get(i, "")
        if value is None:
            cases.at[i, "Ranke grade"] = "skipped"
            cases.at[i, "Ranke score"] = ""
            continue
        g = grade(value, fail_below, pass_from)
        cases.at[i, "Ranke score"] = round(value, 3)
        cases.at[i, "Ranke grade"] = g
        cases.at[i, "Weakest sentence"] = weakest[i][q] if g != "pass" else ""
        label = cases.at[i, "Expert label"]
        if label in ("pass", "fail"):
            cases.at[i, "Matches expert"] = "yes" if g == label else "review" if g == "review" else "no"
    return cases


def _tool_info(run_info, grader, question, fail_below, pass_from, mode, with_input):
    info = dict(run_info)
    info.update({
        "Ranke version": __version__,
        "Ranke mode": mode,
        "Ranke date": date.today().isoformat(),
        "Grader": grader.name,
        "Grader model revision": grader.revision,
        "Question used": question,
        "Fail below": str(fail_below),
        "Pass from": str(pass_from),
        "Tuned": "no",
        "User input sent to Laya": "yes" if with_input else "no",
    })
    return info


def run_test(cases, run_info, grader, fail_below, pass_from, progress=None, with_input=False):
    cases = assign_splits(cases)
    labeled = cases["Expert label"].isin(["pass", "fail"])
    if not labeled.any():
        raise ValueError("No rows have an Expert label of pass or fail, so there is nothing to test Laya against.")
    questions, opposite = questions_from(run_info)
    asked = questions + ([opposite] if opposite else [])
    scores, weakest, notes = _score_rows(grader, cases, labeled, asked, progress, with_input)

    choose = labeled & (cases["Split"] == "dev")
    if not choose.any():
        choose = labeled
    comparison = []
    for q, text in enumerate(questions):
        idx = [i for i in cases.index[choose] if scores[i][q] is not None]
        grades = [grade(scores[i][q], fail_below, pass_from) for i in idx]
        stats = agreement(cases.loc[idx, "Expert label"], grades)
        comparison.append({"Question": text, **stats, "balanced": balanced(stats)})
    best = max(range(len(questions)), key=lambda q: -1 if math.isnan(comparison[q]["balanced"]) else comparison[q]["balanced"])

    cases = _apply(cases, labeled, scores, weakest, notes, best, fail_below, pass_from)
    report = test_report(cases)
    report["questions"] = pd.DataFrame(comparison)
    report["chosen"] = questions[best]

    flip = None
    if opposite:
        idx = [i for i in cases.index[labeled] if scores[i][best] is not None][:FLIP_SAMPLE]
        flip = correlation([scores[i][best] for i in idx], [scores[i][len(questions)] for i in idx])
        if not math.isnan(flip) and flip > FLIP_WARN:
            report["warnings"].append(
                f"Flip check: asking the opposite question gave nearly the same answers (correlation {flip:.2f}). "
                "Laya is not reading the wording closely, so keep questions in the positive form."
            )
    report["flip"] = flip
    report["weak_spots"] = weak_spots(report["dimensions"])
    info = _tool_info(run_info, grader, questions[best], fail_below, pass_from, "test", with_input)
    return cases, info, report


def summary_sheet(report, fail_below, pass_from):
    rows = [("Question used", report["chosen"]), ("Fail below", fail_below), ("Pass from", pass_from)]
    for split, stats in report["by_split"].items():
        for key, value in stats.items():
            rows.append((f"{split}: {key}", value))
    if report["flip"] is not None:
        rows.append(("Flip check correlation", report["flip"]))
    rows += [("Weak spot", s) for s in report["weak_spots"]]
    rows += [("Warning", w) for w in report["warnings"]]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def read_test_sheet(sheet):
    if sheet is None or sheet.empty:
        return None
    items = dict(zip(sheet["Item"], sheet["Value"]))
    return {
        "question": items.get("Question used") or None,
        "fail_below": float(items["Fail below"]) if items.get("Fail below") else None,
        "pass_from": float(items["Pass from"]) if items.get("Pass from") else None,
        "weak_spots": [v for k, v in zip(sheet["Item"], sheet["Value"]) if k == "Weak spot"],
        "catch_rate": items.get("final: catch rate") or items.get("all: catch rate"),
        "few_cases": any(k == "Warning" and "Aim for at least" in str(v) for k, v in zip(sheet["Item"], sheet["Value"])),
    }


def run_eval(cases, run_info, grader, fail_below, pass_from, previous=None, progress=None, with_input=False):
    unlabeled = cases["Expert label"] == ""
    if not unlabeled.any():
        raise ValueError("Every row has an Expert label. eval grades rows without one; use test for labeled rows.")
    warnings = []
    question = None
    if previous:
        question = previous["question"]
        fail_below = previous["fail_below"] if previous["fail_below"] is not None else fail_below
        pass_from = previous["pass_from"] if previous["pass_from"] is not None else pass_from
        for spot in previous["weak_spots"]:
            warnings.append(f"From test: {spot}. The real made-up rate there may be higher than shown.")
        if previous["few_cases"]:
            warnings.append("From test: Laya was checked on too few expert-labeled cases, so treat these numbers as rough.")
    else:
        warnings.append("Laya hasn't been checked against your expert yet (run ranke test first), so these results may not be reliable.")
    if not question:
        question = questions_from(run_info)[0][0]
    scores, weakest, notes = _score_rows(grader, cases, unlabeled, [question], progress, with_input)
    cases = _apply(cases, unlabeled, scores, weakest, notes, 0, fail_below, pass_from)
    report = eval_report(cases[unlabeled])
    report["warnings"] = warnings
    report["question"] = question
    info = _tool_info(run_info, grader, question, fail_below, pass_from, "eval", with_input)
    return cases, info, report, fail_below, pass_from
