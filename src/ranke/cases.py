import random
from pathlib import Path

import pandas as pd

CASE_COLUMNS = [
    "ID",
    "Feature",
    "Scenario",
    "Persona",
    "User input",
    "Context",
    "AI output",
    "Expert label",
    "Expert note",
    "Split",
    "Made-up type",
    "Difficulty",
    "Pair ID",
    "Prompt version",
    "Guardrail triggered",
]

RESULT_COLUMNS = ["Ranke score", "Ranke grade", "Weakest sentence", "Matches expert", "Ranke note"]

DIMENSIONS = ["Feature", "Scenario", "Persona", "Made-up type", "Difficulty", "Prompt version"]

RUN_INFO_FIELDS = [
    "App model",
    "Model settings",
    "System prompt",
    "Prompt version",
    "Guardrails",
    "Retrieval settings",
    "Dataset version",
    "Date",
    "Labeled by",
    "Laya question",
    "Laya question 2",
    "Laya question 3",
    "Laya question (opposite)",
]

LABELS = {"pass", "fail", "unsure"}
SPLITS = ("tuning", "dev", "final")

EXAMPLE_CONTEXT = (
    "Alex Kim, Software Engineer Intern, Acme Corp (Jun-Aug 2025)\n"
    "- Built a REST API in Python and Flask used by 3 internal teams\n"
    "- Reduced database query time by 30% using indexing in PostgreSQL\n"
    "- Member of a 5-person team that migrated services to AWS\n"
    "Skills: Python, Flask, PostgreSQL, AWS (basic), Git"
)

EXAMPLE_ROWS = [
    {
        "ID": "001",
        "Feature": "Experience bullets",
        "Scenario": "Job matches existing skills",
        "Persona": "New grad",
        "User input": "Backend Engineer. Needs Python, PostgreSQL, AWS, Docker.",
        "Context": EXAMPLE_CONTEXT,
        "AI output": "Built a Python/Flask REST API used by 3 internal teams. Cut PostgreSQL query time by 30% through indexing.",
        "Expert label": "pass",
        "Expert note": "Reworded, but every claim is in the base resume.",
        "Pair ID": "P1",
    },
    {
        "ID": "002",
        "Feature": "Skills section",
        "Scenario": "Job asks for a skill you don't have",
        "Persona": "New grad",
        "User input": "Backend Engineer. Needs Python, PostgreSQL, AWS, Docker.",
        "Context": EXAMPLE_CONTEXT,
        "AI output": "Skills: Python, Flask, PostgreSQL, AWS, Docker, Git",
        "Expert label": "fail",
        "Expert note": "Docker isn't in the base resume.",
        "Made-up type": "Invented thing",
        "Difficulty": "Obvious",
        "Pair ID": "P2",
    },
]


class CasesError(Exception):
    pass


def write_template(path):
    path = Path(path)
    cases = pd.DataFrame(EXAMPLE_ROWS, columns=CASE_COLUMNS)
    run_info = pd.DataFrame({"Field": RUN_INFO_FIELDS, "Value": [""] * len(RUN_INFO_FIELDS)})
    if path.suffix.lower() == ".csv":
        cases.to_csv(path, index=False)
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        cases.to_excel(writer, sheet_name="Cases", index=False)
        run_info.to_excel(writer, sheet_name="Run info", index=False)


def _clean(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in CASE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    for col in CASE_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df["Expert label"] = df["Expert label"].str.lower()
    df["Split"] = df["Split"].str.lower()
    bad = sorted(set(df["Expert label"]) - LABELS - {""})
    if bad:
        raise CasesError(f"Expert label must be pass, fail, unsure or empty. Found: {', '.join(bad)}")
    bad = sorted(set(df["Split"]) - set(SPLITS) - {""})
    if bad:
        raise CasesError(f"Split must be tuning, dev, final or empty. Found: {', '.join(bad)}")
    if (df["ID"] == "").any():
        df.loc[df["ID"] == "", "ID"] = [f"row-{i + 2}" for i in df.index[df["ID"] == ""]]
    return df


def load(path):
    path = Path(path)
    if not path.exists():
        raise CasesError(f"File not found: {path}")
    sheets = {}
    if path.suffix.lower() == ".csv":
        cases = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
        if "Cases" not in sheets:
            raise CasesError('The Excel file needs a sheet called "Cases".')
        cases = sheets["Cases"]
    missing = [c for c in ("Context", "AI output") if c not in [str(x).strip() for x in cases.columns]]
    if missing:
        raise CasesError(f"Missing required column(s): {', '.join(missing)}")
    run_info = {}
    if "Run info" in sheets:
        for _, row in sheets["Run info"].iterrows():
            field = str(row.get("Field", "")).strip()
            if field:
                run_info[field] = str(row.get("Value", "")).strip()
    previous_test = sheets.get("Ranke test")
    return _clean(cases), run_info, previous_test


def assign_splits(cases, seed=7):
    cases = cases.copy()
    labeled = cases["Expert label"].isin(["pass", "fail"]) & (cases["Split"] == "")
    if not labeled.any():
        return cases
    groups = cases.loc[labeled].apply(lambda r: r["Pair ID"] or r["ID"], axis=1)
    keys = sorted(set(groups))
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_tuning = round(n * 0.15)
    n_dev = round(n * 0.425)
    choice = {}
    for i, k in enumerate(keys):
        choice[k] = "tuning" if i < n_tuning else "dev" if i < n_tuning + n_dev else "final"
    cases.loc[labeled, "Split"] = groups.map(choice)
    return cases


def output_paths(path, mode):
    path = Path(path)
    stem = path.stem.removesuffix("_results")
    suffix = ".csv" if path.suffix.lower() == ".csv" else ".xlsx"
    return path.with_name(f"{stem}_results{suffix}"), path.with_name(f"{stem}_{mode}_report.html")


def save(path, cases, run_info, extra_sheets):
    path = Path(path)
    cases = cases.reindex(columns=list(dict.fromkeys(CASE_COLUMNS + list(cases.columns))), fill_value="")
    columns = CASE_COLUMNS + [c for c in RESULT_COLUMNS if c in cases.columns]
    columns += [c for c in cases.columns if c not in columns]
    cases = cases[columns]
    if path.suffix.lower() == ".csv":
        cases.to_csv(path, index=False)
        return
    info = pd.DataFrame({"Field": list(run_info.keys()), "Value": list(run_info.values())})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        cases.to_excel(writer, sheet_name="Cases", index=False)
        info.to_excel(writer, sheet_name="Run info", index=False)
        for name, df in extra_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
