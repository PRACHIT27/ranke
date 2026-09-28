# Ranke

**Check whether your AI made things up.**

Ranke reads your AI's outputs and checks that everything in them is backed by the source the AI was given. For example: did your resume tailor add a skill that isn't in the base resume? Did your chatbot quote a number that isn't in the document?

It's named after Leopold von Ranke, the historian who insisted every claim be checked against the original sources.

- **Free and private.** The grader, [Laya](https://huggingface.co/convaiinnovations/laya), runs on your own computer. Your data isn't sent anywhere.
- **Checked against your expert.** Before trusting Ranke's grades, you test it against labels from a person who knows your field.
- **Excel in, Excel out.** Fill in a spreadsheet, run one command, and get the same spreadsheet back with grades, plus a report card.

## How it works

Ranke does two jobs:

| Command | Question it answers | Uses rows that... |
|---|---|---|
| `ranke test` | Can I trust Laya's grades? | **have** an expert label |
| `ranke eval` | How often does my AI make things up? | **don't have** an expert label |

Run `test` first. Once you know how far Laya can be trusted, `eval` grades the rest of your outputs without a human.

## Install

You need Python 3.10 or newer and about 3 GB of free disk space. No graphics card is needed.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ranke
```

The first run downloads the Laya model (about 810 MB) from Hugging Face.

## Quick start

```bash
ranke template cases.xlsx
```

Open `cases.xlsx` and fill in one row per case. Then:

```bash
ranke test cases.xlsx
```

```bash
ranke eval cases_results.xlsx
```

You get:

- `cases_results.xlsx`: your file with Ranke's grades added
- `cases_test_report.html` and `cases_eval_report.html`: report cards you can open in any browser

## Filling in the spreadsheet

### The "Cases" sheet (one row per case)

| Column | Required? | What to put in it |
|---|---|---|
| ID | no | Any name for the row |
| Feature | no | Which part of your app, like "Skills section" |
| Scenario | no | The situation, like "Job asks for a skill you don't have" |
| Persona | no | The type of user, like "New grad" |
| User input | no | What the user asked. Not shown to the grader unless you use `--with-input`. |
| **Context** | **yes** | The source the AI was given, like the base resume |
| **AI output** | **yes** | What your AI wrote |
| Expert label | for `test` | `pass` (nothing made up), `fail` (something made up), or `unsure` |
| Expert note | no | Why the expert chose that label |
| Split | no | `tuning`, `dev`, or `final`. Leave it empty and Ranke fills it in. |
| Made-up type | no | Like "Invented thing", "Wrong number", "Overclaiming" |
| Difficulty | no | Like "Obvious", "Medium", "Subtle" |
| Pair ID | no | The same ID for a clean row and its made-up twin |
| Prompt version | no | Which version of your prompt produced this output |
| Guardrail triggered | no | Whether a safety rule changed the output |

The optional columns are used to break down results in the report, so the more you fill in, the more you learn.

### The "Run info" sheet (filled in once)

Details about your AI: model, prompt, guardrails, dataset version, and who labeled the cases. You can also try up to three different wordings of the question Laya is asked (`Laya question`, `Laya question 2`, `Laya question 3`). `ranke test` picks the one that agrees best with your expert.

## Reading the report

**Test report:**

- **Catch rate:** of the cases your expert marked `fail`, how many Ranke also failed. This is the most important number.
- **Correct-pass rate:** of the cases your expert marked `pass`, how many Ranke also passed.
- These two are always shown separately. A grader that passes everything can look "95% accurate" while catching nothing.
- **Where Laya is weak:** tags (like a made-up type) where it misses a lot.
- **Disagreements:** every case where Ranke and your expert differ. Read these. It's the most useful part.

**Eval report:**

- **Made-up rate:** how often your AI made something up, overall and broken down by each tag.
- Warnings carried over from `test`, like "misses overclaiming, so the real rate may be higher."

### Grades

Every row gets a score from 0 to 1 (how well supported it is) and a grade:

| Score | Grade |
|---|---|
| below 0.3 | `fail` |
| 0.3 to 0.9 | `review` (a person should look) |
| 0.9 and above | `pass` |

Change these cutoffs with `--fail-below` and `--pass-from`.

## Good test data

Following [Hamel Husain's guide](https://hamel.dev/blog/posts/llm-judge/):

- Start with one main expert, and have them label at least **30 passes and 30 fails**. Ranke warns you if you have fewer.
- Cover different features, scenarios, and personas.
- **Plant made-up content on purpose:** take an output you know is correct, change one thing (add a skill, change a number), and label it `fail`. Give both rows the same Pair ID.
- Include easy, medium, and subtle cases.
- Only use Ranke once you know made-up content is a real problem for your AI. Look at your data first.

## Run it on GitHub

Ranke works as a GitHub Action that runs **in your own repo**. Your data stays in your account. Copy [examples/ranke-workflow.yml](examples/ranke-workflow.yml) into `.github/workflows/` in your repo:

```yaml
- uses: PRACHIT27/ranke@v1
  with:
    file: cases.xlsx
    mode: both
```

It runs on GitHub's free machines (no graphics card needed). Results are saved as a download on the run page for 3 days, and the run summary shows counts only, never your data.

**Use a private repo for real data.** Anything in a public repo is visible to everyone.

## Using Jev (optional, your own key)

[Jev](https://docs.typesafe.ai/) by TypeSafe is a paid grader that can read much longer text than Laya. If you have a key:

```bash
export TYPESAFE_API_KEY=your-key
ranke test cases.xlsx --grader auto
```

- `--grader laya`: Laya only (default)
- `--grader auto`: Laya, but rows with long context go to Jev
- `--grader jev`: Jev for everything

**With `jev` or `auto`, your rows are sent to TypeSafe's servers.** On GitHub, store the key as a secret and pass it as `typesafe-api-key: ${{ secrets.TYPESAFE_API_KEY }}`.

## Privacy

| When | What leaves your machine |
|---|---|
| Grading with Laya | nothing |
| First run | the Laya model is downloaded from Hugging Face |
| `--offline` | nothing at all (the model must already be downloaded) |
| Grading with Jev | your rows go to TypeSafe |
| On GitHub | your files are stored in your GitHub account |

Ranke turns off Hugging Face usage tracking and locks the exact Laya model version, so the model can't change underneath you.

## What Laya is good and bad at

We tested Laya on 600 examples from the public [HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) dataset without any tuning:

- It caught about **half** of the made-up answers and passed about **9 in 10** correct ones.
- When it says something is **not** supported, it's usually right. When it says something **is** supported, it's often too confident.
- It reads only about 350 words at a time, so Ranke splits long outputs into sentences and long context into overlapping pieces.
- It ignores the word "not" in questions, so always ask in the positive form ("Is everything supported?"). `ranke test` checks this for you.

That's why `ranke test` exists: check it against your own expert before relying on it.

## Limits

- Ranke checks only against the context you give it. It can't tell whether something is true in the real world.
- Laya gives no reasons, only a score. Ranke shows the **weakest sentence** in each row to point you to the likely problem.
- A claim that needs two far-apart parts of the context together may be wrongly failed.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Ranke is not endorsed by Convai Innovations or TypeSafe AI.
