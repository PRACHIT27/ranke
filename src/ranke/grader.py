import os
import re
import warnings

MODEL_REPO = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
MODEL_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
TOKEN_LIMIT = 512

DEFAULT_QUESTION = "Is every claim in the OUTPUT fully supported by the CONTEXT?"
DEFAULT_OPPOSITE = "Does the OUTPUT contain any claim that is not supported by the CONTEXT?"

CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
REQUEST_WORDS = 60

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")


def split_sentences(text):
    pieces = []
    for line in str(text).splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        pieces.extend(p.strip() for p in _SENTENCE_END.split(line) if p.strip())
    return [p for p in pieces if len(p) >= 3] or ([str(text).strip()] if str(text).strip() else [])


def chunk_words(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = str(text).split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return chunks
        start += size - overlap


def _trim(text, words):
    parts = str(text).split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


class LayaGrader:
    name = "Laya (English)"
    revision = MODEL_REVISION
    context_words = CHUNK_WORDS

    def __init__(self, offline=False, device=None):
        os.environ.setdefault("USE_TF", "0")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        from huggingface_hub import snapshot_download
        import laya

        path = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=MODEL_FILES)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid temperatures")
            self._agent = laya.load(path, device=device)

    def support(self, context, output, request, questions):
        state = {"context": context, "output": output}
        if request:
            state = {"context": context, "question": _trim(request, REQUEST_WORDS), "output": output}
        asked = {f"q{i}": {"type": "noul", "instructions": q} for i, q in enumerate(questions)}
        result = self._agent.predict(state, asked)
        scores = [float(result["answers"][f"q{i}"]["noul"]) for i in range(len(questions))]
        return scores, int(result.get("usage", {}).get("input_tokens", 0))


def score_row(grader, context, output, request, questions):
    sentences = split_sentences(output)
    chunks = chunk_words(context, size=grader.context_words) if grader.context_words else [context]
    best = [[0.0] * len(sentences) for _ in questions]
    max_tokens = 0
    for s, sentence in enumerate(sentences):
        for chunk in chunks:
            scores, tokens = grader.support(chunk, sentence, request, questions)
            max_tokens = max(max_tokens, tokens)
            for q, value in enumerate(scores):
                best[q][s] = max(best[q][s], value)
    results = []
    for q in range(len(questions)):
        weakest = min(range(len(sentences)), key=lambda s: best[q][s])
        results.append((best[q][weakest], sentences[weakest]))
    return {
        "results": results,
        "sentences": len(sentences),
        "chunks": len(chunks),
        "cut_off": max_tokens > TOKEN_LIMIT,
    }


JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_KEY_ENV = "TYPESAFE_API_KEY"
JEV_RETRY_STATUSES = {429, 529}


class JevError(Exception):
    pass


class JevGrader:
    name = "Jev (TypeSafe)"
    context_words = None

    def __init__(self, model="jev-latest", api_key=None, retries=4, timeout=60):
        self.api_key = api_key or os.environ.get(JEV_KEY_ENV, "")
        if not self.api_key:
            raise JevError(f"Jev needs your own TypeSafe key in the {JEV_KEY_ENV} environment variable.")
        self.model = model
        self.revision = model
        self.retries = retries
        self.timeout = timeout

    def _post(self, body):
        import json
        import time
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(JEV_URL, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in JEV_RETRY_STATUSES and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                if e.code == 401:
                    raise JevError("TypeSafe rejected the API key (401).") from None
                detail = e.read().decode("utf-8", "replace")[:300]
                raise JevError(f"TypeSafe returned {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise JevError(f"Could not reach TypeSafe: {e.reason}") from None

    def support(self, context, output, request, questions):
        state = {"context": context, "output": output}
        if request:
            state = {"context": context, "question": _trim(request, REQUEST_WORDS), "output": output}
        body = {
            "state": state,
            "model": self.model,
            "questions": {f"q{i}": {"type": "noul", "instructions": q} for i, q in enumerate(questions)},
        }
        result = self._post(body)
        self.revision = result.get("model", self.model)
        scores = [float(result["answers"][f"q{i}"]["noul"]) for i in range(len(questions))]
        return scores, 0


class AutoGrader:
    context_words = None

    def __init__(self, laya, jev):
        self.laya = laya
        self.jev = jev
        self.name = f"{laya.name}, long rows to {jev.name}"
        self.revision = f"{laya.revision} / {jev.model}"

    def pick(self, context):
        return self.jev if len(str(context).split()) > CHUNK_WORDS else self.laya


def grade_row(grader, context, output, request, questions):
    used = grader.pick(context) if isinstance(grader, AutoGrader) else grader
    scored = score_row(used, context, output, request, questions)
    scored["grader"] = used.name
    return scored
