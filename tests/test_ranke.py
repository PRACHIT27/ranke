import pandas as pd
import pytest

from ranke import cli
from ranke.cases import CasesError, assign_splits, load, output_paths, write_template
from ranke.grader import DEFAULT_QUESTION, chunk_words, score_row, split_sentences
from ranke.metrics import agreement, grade
from ranke.runner import read_test_sheet, run_eval, run_test, summary_sheet


class FakeGrader:
    name = "fake"
    revision = "test"
    context_words = 180

    def __init__(self, bad_words=("Docker", "Led", "60%")):
        self.bad = bad_words
        self.calls = 0

    def support(self, context, output, request, questions):
        self.calls += 1
        made_up = any(w in output for w in self.bad) or output.split()[0] not in context
        score = 0.05 if made_up else 0.97
        return [score] * len(questions), 100


def test_split_sentences_handles_bullets_and_lines():
    text = "- Built an API. Cut query time by 30%.\n* Led the team\n\n2) Wrote tests"
    assert split_sentences(text) == ["Built an API.", "Cut query time by 30%.", "Led the team", "Wrote tests"]


def test_split_sentences_keeps_decimals_together():
    assert split_sentences("Grew revenue 2.5x in 2024. Then left.") == ["Grew revenue 2.5x in 2024.", "Then left."]


def test_chunk_words_overlaps():
    words = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_words(words, size=180, overlap=40)
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "w140"
    assert chunks[-1].split()[-1] == "w399"
    assert chunk_words("short text") == ["short text"]


def test_score_row_takes_best_chunk_and_weakest_sentence():
    grader = FakeGrader()
    context = "Built an API. " + " ".join(["filler"] * 300) + " Wrote tests daily."
    scored = score_row(grader, context, "Built an API. Led a team.", "", [DEFAULT_QUESTION])
    score, weakest = scored["results"][0]
    assert score == pytest.approx(0.05)
    assert weakest == "Led a team."
    assert scored["chunks"] == 2 and scored["sentences"] == 2


def test_grade_bands():
    assert grade(0.1, 0.3, 0.9) == "fail"
    assert grade(0.5, 0.3, 0.9) == "review"
    assert grade(0.95, 0.3, 0.9) == "pass"


def test_agreement_reports_both_rates_separately():
    stats = agreement(["fail", "fail", "pass", "pass"], ["fail", "pass", "pass", "fail"])
    assert stats["catch rate"] == 0.5
    assert stats["correct-pass rate"] == 0.5
    assert stats["false alarms"] == 0.5


def test_template_round_trip(tmp_path):
    path = tmp_path / "cases.xlsx"
    write_template(path)
    cases, run_info, previous = load(path)
    assert list(cases["Expert label"]) == ["pass", "fail"]
    assert "Laya question" in run_info
    assert previous is None


def test_bad_label_is_rejected(tmp_path):
    path = tmp_path / "cases.csv"
    pd.DataFrame({"Context": ["a"], "AI output": ["b"], "Expert label": ["maybe"]}).to_csv(path, index=False)
    with pytest.raises(CasesError):
        load(path)


def test_splits_keep_pairs_together():
    cases = pd.DataFrame({
        "ID": [str(i) for i in range(40)],
        "Pair ID": [f"P{i // 2}" for i in range(40)],
        "Expert label": ["pass", "fail"] * 20,
        "Split": [""] * 40,
    })
    split = assign_splits(cases)
    assert split.groupby("Pair ID")["Split"].nunique().max() == 1
    assert set(split["Split"]) == {"tuning", "dev", "final"}


def _cases(n_pairs=40, unlabeled=6):
    context = "Built a REST API in Python. Member of a 5-person team. Cut query time by 30%."
    rows = []
    for i in range(n_pairs):
        rows.append({"ID": f"{i}a", "Pair ID": f"P{i}", "Context": context, "AI output": "Built a REST API in Python.",
                     "Expert label": "pass", "Feature": "Experience"})
        rows.append({"ID": f"{i}b", "Pair ID": f"P{i}", "Context": context,
                     "AI output": "Led a 5-person team." if i % 2 else "Skills: Docker.",
                     "Expert label": "fail", "Feature": "Experience" if i % 2 else "Skills",
                     "Made-up type": "Overclaiming" if i % 2 else "Invented thing"})
    for i in range(unlabeled):
        rows.append({"ID": f"u{i}", "Context": context, "AI output": "Cut query time by 60%." if i % 2 else "Built a REST API in Python."})
    rows.append({"ID": "empty", "Context": "", "AI output": "x"})
    return pd.DataFrame(rows).fillna("")


def test_run_test_and_eval_end_to_end():
    from ranke.cases import _clean
    cases = _clean(_cases())
    grader = FakeGrader()
    tested, info, rep = run_test(cases, {}, grader, 0.3, 0.9)
    assert rep["by_split"]["all"]["catch rate"] == 1.0
    assert rep["by_split"]["all"]["correct-pass rate"] == 1.0
    assert info["Question used"] == DEFAULT_QUESTION
    assert rep["flip"] is not None
    sheet = summary_sheet(rep, 0.3, 0.9)
    previous = read_test_sheet(sheet)
    assert previous["question"] == DEFAULT_QUESTION

    evaluated, _, erep, _, _ = run_eval(tested, {}, grader, 0.3, 0.9, previous)
    assert erep["overall"]["rows"] == 6
    assert erep["overall"]["made-up rate"] == pytest.approx(0.5)
    assert erep["skipped"] == 1
    assert evaluated.loc[evaluated["ID"] == "empty", "Ranke grade"].item() == "skipped"


def test_flip_check_warns_when_opposite_matches():
    from ranke.cases import _clean
    _, _, rep = run_test(_clean(_cases()), {}, FakeGrader(), 0.3, 0.9)
    assert any("Flip check" in w for w in rep["warnings"])


def test_eval_without_test_warns():
    from ranke.cases import _clean
    _, _, rep, _, _ = run_eval(_clean(_cases()), {}, FakeGrader(), 0.3, 0.9)
    assert any("run ranke test first" in w for w in rep["warnings"])


def test_cli_end_to_end(tmp_path, monkeypatch):
    path = tmp_path / "cases.xlsx"
    from ranke.cases import save
    save(path, _cases(), {"Laya question": ""}, {})
    monkeypatch.setattr(cli, "_grader", lambda args: FakeGrader())
    assert cli.main(["test", str(path)]) == 0
    results, test_report = output_paths(path, "test")
    assert results.exists() and test_report.exists()
    assert cli.main(["eval", str(results)]) == 0
    _, eval_report = output_paths(path, "eval")
    sheets = pd.read_excel(results, sheet_name=None)
    assert {"Cases", "Run info", "Ranke test"} <= set(sheets)
    assert "Ranke test report" in test_report.read_text(encoding="utf-8")
    assert "Ranke eval report" in eval_report.read_text(encoding="utf-8")


def test_jev_grader_needs_key(monkeypatch):
    from ranke.grader import JevError, JevGrader
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError):
        JevGrader()


def test_jev_grader_sends_documented_request(monkeypatch):
    import json
    from ranke.grader import JEV_URL, JevGrader
    sent = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"model": "jev-1.13.0", "answers": {"q0": {"type": "noul", "noul": 0.2}},
                               "usage": {"input_tokens": 10, "output_tokens": 1}}).encode()

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["auth"] = req.get_header("Authorization")
        sent["body"] = json.loads(req.data)
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    grader = JevGrader(api_key="k")
    scores, _ = grader.support("ctx", "out", "", [DEFAULT_QUESTION])
    assert scores == [0.2]
    assert sent["url"] == JEV_URL and sent["auth"] == "Bearer k"
    assert sent["body"]["model"] == "jev-latest"
    assert sent["body"]["questions"]["q0"] == {"type": "noul", "instructions": DEFAULT_QUESTION}
    assert grader.revision == "jev-1.13.0"


def test_auto_grader_sends_only_long_context_to_jev():
    from ranke.grader import AutoGrader, grade_row

    class Named(FakeGrader):
        def __init__(self, name, context_words):
            super().__init__()
            self.name = name
            self.model = name
            self.context_words = context_words

    auto = AutoGrader(Named("laya", 180), Named("jev", None))
    short = grade_row(auto, "Built an API.", "Built an API.", "", [DEFAULT_QUESTION])
    long = grade_row(auto, "Built an API. " + "filler " * 300, "Built an API.", "", [DEFAULT_QUESTION])
    assert short["grader"] == "laya" and long["grader"] == "jev"
    assert long["chunks"] == 1
