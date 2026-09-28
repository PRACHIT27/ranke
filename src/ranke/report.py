import html
import math

import pandas as pd

STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:1000px;color:#1f2328;padding:0 1rem}
h1{margin-bottom:.2rem}h2{margin-top:2rem;border-bottom:1px solid #d0d7de;padding-bottom:.3rem}
.sub{color:#59636e;margin-top:0}
.tiles{display:flex;gap:1rem;flex-wrap:wrap}
.tile{border:1px solid #d0d7de;border-radius:8px;padding:.8rem 1rem;min-width:170px}
.tile b{display:block;font-size:1.8rem}
.warn{background:#fff8c5;border:1px solid #d4a72c;border-radius:6px;padding:.5rem .8rem;margin:.4rem 0}
table{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:.92rem}
th,td{border:1px solid #d0d7de;padding:.35rem .55rem;text-align:left;vertical-align:top}
th{background:#f6f8fa}
td.num{text-align:right;white-space:nowrap}
"""


def _fmt(value, key=""):
    if isinstance(value, float):
        if math.isnan(value):
            return "–"
        if "rate" in key or "review" in key or "missed" in key or "alarms" in key or "supported" in key or key == "balanced":
            return f"{value:.0%}"
        return f"{value:.3f}"
    return html.escape(str(value))


def _table(df, first_cols=None):
    if df is None or df.empty:
        return "<p>No data.</p>"
    cols = list(df.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cls = ' class="num"' if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
            cells.append(f"<td{cls}>{_fmt(v, str(c))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _tile(label, value, key):
    return f'<div class="tile">{html.escape(label)}<b>{_fmt(value, key)}</b></div>'


def _warnings(items):
    return "".join(f'<div class="warn">{html.escape(w)}</div>' for w in items)


def _page(title, subtitle, parts):
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{STYLE}</style></head><body><h1>{html.escape(title)}</h1>"
        f"<p class='sub'>{html.escape(subtitle)}</p>{''.join(parts)}</body></html>"
    )


def test_page(report, info, cases):
    split = "final" if "final" in report["by_split"] else "all"
    main = report["by_split"].get(split, {})
    parts = [
        f"<h2>Can you trust Laya? ({split} split)</h2>",
        '<div class="tiles">',
        _tile("Catch rate", main.get("catch rate", float("nan")), "catch rate"),
        _tile("Correct-pass rate", main.get("correct-pass rate", float("nan")), "correct-pass rate"),
        _tile("False alarms", main.get("false alarms", float("nan")), "false alarms"),
        _tile("Fails sent to review", main.get("fails sent to review", float("nan")), "fails sent to review"),
        "</div>",
        _warnings(report["warnings"]),
        "<h2>By split</h2>",
        _table(pd.DataFrame([{"split": k, **v} for k, v in report["by_split"].items()])),
        "<h2>Question wordings (dev split)</h2>",
        f"<p>Used for grades: <b>{html.escape(report['chosen'])}</b></p>",
        _table(report["questions"]),
    ]
    if report["weak_spots"]:
        parts += ["<h2>Where Laya is weak</h2><ul>"] + [f"<li>{html.escape(s)}</li>" for s in report["weak_spots"]] + ["</ul>"]
    for dim, table in report["dimensions"].items():
        parts += [f"<h2>By {html.escape(dim.lower())}</h2>", _table(table)]
    disagree = cases[cases["Matches expert"].isin(["no", "review"])]
    parts += [
        f"<h2>Disagreements with your expert ({len(disagree)})</h2>",
        _table(disagree[["ID", "Expert label", "Ranke grade", "Ranke score", "Weakest sentence", "Expert note"]]),
    ]
    return _page("Ranke test report", _subtitle(info), parts)


def eval_page(report, info, cases):
    overall = report["overall"]
    parts = [
        "<h2>How is your AI doing?</h2>",
        '<div class="tiles">',
        _tile("Made-up rate", overall["made-up rate"], "made-up rate"),
        _tile("Needs review", overall["needs review"], "needs review"),
        _tile("Looks supported", overall["looks supported"], "looks supported"),
        _tile("Rows graded", overall["rows"], ""),
        "</div>",
        _warnings(report["warnings"]),
    ]
    if report["skipped"]:
        parts.append(_warnings([f"{report['skipped']} row(s) skipped. See the Ranke note column."]))
    for dim, table in report["dimensions"].items():
        parts += [f"<h2>By {html.escape(dim.lower())}</h2>", _table(table)]
    flagged = cases[cases["Ranke grade"].isin(["fail", "review"])]
    parts += [
        f"<h2>Flagged rows ({len(flagged)})</h2>",
        _table(flagged[["ID", "Ranke grade", "Ranke score", "Weakest sentence", "Ranke note"]]),
    ]
    return _page("Ranke eval report", _subtitle(info), parts)


def _subtitle(info):
    bits = [f"Ranke {info.get('Ranke version', '')}", info.get("Ranke date", ""), info.get("Grader", "")]
    if info.get("Prompt version"):
        bits.append(f"prompt {info['Prompt version']}")
    if info.get("Dataset version"):
        bits.append(f"dataset {info['Dataset version']}")
    return " · ".join(b for b in bits if b)


def write(path, page):
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
