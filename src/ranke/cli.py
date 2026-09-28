import argparse
import sys
import time

from . import __version__, report
from .cases import CasesError, load, output_paths, save, write_template
from .grader import JevError
from .runner import read_test_sheet, run_eval, run_test, summary_sheet

DEFAULT_FAIL_BELOW = 0.3
DEFAULT_PASS_FROM = 0.9


def _progress():
    start = time.time()

    def show(done, total):
        if done == total or done % 25 == 0:
            print(f"  graded {done}/{total} rows ({time.time() - start:.0f}s)", flush=True)

    return show


def _pct(value):
    return "n/a" if value != value else f"{value:.0%}"


def _grader(args):
    from .grader import AutoGrader, JevGrader, LayaGrader
    jev = None
    if args.grader in ("jev", "auto"):
        jev = JevGrader(model=args.jev_model)
        where = "every row" if args.grader == "jev" else "rows with long context"
        print(f"Note: {where} will be sent to TypeSafe's servers for grading.", flush=True)
        if args.grader == "jev":
            return jev
    print("Loading Laya...", flush=True)
    laya = LayaGrader(offline=args.offline)
    return AutoGrader(laya, jev) if jev else laya


def cmd_template(args):
    write_template(args.path)
    print(f"Template written to {args.path}")


def cmd_test(args):
    cases, run_info, _ = load(args.file)
    cases, info, rep = run_test(cases, run_info, _grader(args), args.fail_below, args.pass_from, _progress(), args.with_input)
    out, html_path = output_paths(args.file, "test")
    save(out, cases, info, {"Ranke test": summary_sheet(rep, args.fail_below, args.pass_from)})
    report.write(html_path, report.test_page(rep, info, cases))
    split = "final" if "final" in rep["by_split"] else "all"
    stats = rep["by_split"][split]
    print(f"Tested {rep['rows']} labeled rows.")
    print(f"  {split} split: catch rate {_pct(stats['catch rate'])}, correct-pass rate {_pct(stats['correct-pass rate'])}")
    print(f"  {len(rep['warnings'])} warning(s), {len(rep['weak_spots'])} weak spot(s)")
    print(f"Results: {out}\nReport:  {html_path}")


def cmd_eval(args):
    cases, run_info, previous_sheet = load(args.file)
    previous = read_test_sheet(previous_sheet)
    cases, info, rep, fail_below, pass_from = run_eval(
        cases, run_info, _grader(args), args.fail_below, args.pass_from, previous, _progress(), args.with_input
    )
    out, html_path = output_paths(args.file, "eval")
    extra = {"Ranke test": previous_sheet} if previous_sheet is not None else {}
    save(out, cases, info, extra)
    report.write(html_path, report.eval_page(rep, info, cases[cases["Expert label"] == ""]))
    o = rep["overall"]
    print(f"Evaluated {o['rows']} rows: made-up {_pct(o['made-up rate'])}, needs review {_pct(o['needs review'])}")
    print(f"  {len(rep['warnings'])} warning(s), {rep['skipped']} skipped")
    print(f"Results: {out}\nReport:  {html_path}")


def build_parser():
    p = argparse.ArgumentParser(prog="ranke", description="Check whether your AI made things up.")
    p.add_argument("--version", action="version", version=f"ranke {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("template", help="write a blank cases file to fill in")
    t.add_argument("path", nargs="?", default="cases.xlsx")
    t.set_defaults(func=cmd_template)

    for name, func, text in (
        ("test", cmd_test, "check Laya against your expert's labels"),
        ("eval", cmd_eval, "grade your AI's outputs on rows without labels"),
    ):
        c = sub.add_parser(name, help=text)
        c.add_argument("file")
        c.add_argument("--fail-below", type=float, default=DEFAULT_FAIL_BELOW)
        c.add_argument("--pass-from", type=float, default=DEFAULT_PASS_FROM)
        c.add_argument("--offline", action="store_true", help="never contact the internet (model must already be downloaded)")
        c.add_argument("--with-input", action="store_true", help="also show the grader the User input, for short answers that need their question")
        c.add_argument("--grader", choices=["laya", "jev", "auto"], default="laya",
                       help="laya (free, local), jev (your own TypeSafe key), or auto (Laya, long rows to Jev)")
        c.add_argument("--jev-model", default="jev-latest")
        c.set_defaults(func=func)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "fail_below", 0) > getattr(args, "pass_from", 1):
        print("--fail-below must not be higher than --pass-from", file=sys.stderr)
        return 2
    try:
        args.func(args)
    except (CasesError, JevError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
