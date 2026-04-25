"""Re-run the held-out 157-test interpretability suite with Claude Haiku as
the LLM evaluator (and a slightly perturbed prompt), to test sensitivity of
the agent-graded interpretability score to evaluator family beyond GPT-5.4.

This script mirrors `evaluate_new_generalization_gpt_5_4.py`:
  - Loads the same 16 baselines + 25 evolved models from the apr9 main run
  - Runs the same set of 157 NEW interpretability tests (`ALL_NEW_TESTS`)
  - Saves results to `new_results_claude_haiku/` in the same CSV layout

The two evaluator-side differences from the GPT-5.4 script:
  1. The LLM is Claude Haiku 4.5 (Anthropic API, with a thin wrapper that
     exposes the same `llm(prompt, max_completion_tokens=..., stop=...)`
     callable interface used by `interp_eval.ask_llm`).
  2. The prompt prefix in `ask_llm` is perturbed (re-worded) to test whether
     the LLM-graded score is sensitive to the exact phrasing of the prompt
     as well as to the evaluator model.

Usage:
  ANTHROPIC_API_KEY=sk-ant-... uv run \
      generalization_experiments/evaluate_new_generalization_claude_haiku.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Memory, Parallel, delayed

# Add src/ paths used by the GPT-5.4 script.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "evolve" / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parent / "evolve"))
sys.path.insert(0, str(SCRIPT_DIR))

from performance_eval import (
    MAX_SAMPLES, MAX_FEATURES, MIN_SAMPLES, MIN_FEATURES, SUBSAMPLE_SEED,
    subsample_dataset, compute_rank_scores, OVERALL_CSV_COLS,
)
import interp_eval  # we monkey-patch interp_eval.ask_llm below
from interp_eval import get_model_str, _safe_clone

# Pull the same 157 test functions used for the GPT-5.4 evaluation.
from evaluate_new_generalization import (
    ALL_NEW_TESTS,
    _ALL_NEW_TEST_FNS,
)

# Reuse the apr9 baselines + evolved models. We load these by re-using the
# existing GPT-5.4 evaluation script's BASELINE_DEFS and load_all_models()
# (the latter walks LIB_DIR for evolved-model python files).
import evaluate_new_generalization_gpt_5_4 as gpt54_mod
from evaluate_new_generalization_gpt_5_4 import (
    BASELINE_DEFS, BASELINE_DESCRIPTIONS, load_all_models,
)

# Use the apr9 evolved-model lib as the source if the local LIB_DIR is empty.
APR9_LIB = (SCRIPT_DIR.parent / "result_libs"
            / "apr9-claude-effort=medium-main-result"
            / "interpretable_regressors_lib" / "success")
if not gpt54_mod.LIB_DIR.exists() or not list(gpt54_mod.LIB_DIR.glob("*.py")):
    gpt54_mod.LIB_DIR = APR9_LIB
    print(f"  Using apr9 lib at {APR9_LIB}")

RESULTS_DIR = SCRIPT_DIR / "new_results_claude_haiku"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Claude Haiku wrapper with a callable interface matching imodelsx LLMs.
# ---------------------------------------------------------------------------

CLAUDE_MODEL = os.environ.get("CLAUDE_EVAL_MODEL", "claude-haiku-4-5")


class ClaudeHaiku:
    """Thin wrapper exposing the imodelsx LLM call signature.

    Defaults to invoking the `claude` CLI in print mode (`claude -p`), which
    uses the user's Max subscription OAuth credentials --- no Anthropic API
    key required. If `ANTHROPIC_API_KEY` is set in env, falls back to the
    Anthropic SDK (faster but billed against the API).

    Call as:  llm(prompt, max_completion_tokens=..., stop=[...])
    """

    SYSTEM_PROMPT = (
        "You are a concise model-interpretation assistant. "
        "Answer the user's question about a printed regression model directly. "
        "Use ONLY the information present in the printed model. "
        "If the answer is a number, just give the number. "
        "If a feature name is requested, just give the feature name."
    )

    def __init__(self, model: str = CLAUDE_MODEL):
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.mode = "api"
                return
            except ImportError:
                pass
        # CLI fallback (Max subscription via OAuth)
        import shutil
        if not shutil.which("claude"):
            raise RuntimeError(
                "Neither ANTHROPIC_API_KEY nor `claude` CLI is available."
            )
        self.mode = "cli"

    def _call_api(self, prompt, max_tokens, stop, temperature):
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop or [],
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))

    def _call_cli(self, prompt, max_tokens, stop, temperature):
        import subprocess
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "text",
            "--max-turns", "1",
            "--system-prompt", self.SYSTEM_PROMPT,
            "--disallowed-tools",
            "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,TodoWrite,Task,NotebookEdit",
        ]
        out = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {out.stderr[:300]}")
        text = out.stdout.strip()
        # Apply stop sequences manually (CLI doesn't support stop_sequences).
        for s in stop or []:
            if s in text:
                text = text.split(s)[0]
        return text

    def __call__(self, prompt, max_completion_tokens=200, stop=None,
                 temperature=0, **kwargs):
        max_retries = 8
        last_err = None
        for attempt in range(max_retries):
            try:
                if self.mode == "api":
                    text = self._call_api(prompt, max_completion_tokens, stop, temperature)
                else:
                    text = self._call_cli(prompt, max_completion_tokens, stop, temperature)
                if not text or not text.strip():
                    # Empty response -> treat as transient and retry instead of
                    # caching nothing.
                    raise RuntimeError("empty response from Claude")
                return text
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                rate_limited = any(kw in msg for kw in
                                   ("rate", "quota", "limit", "5h",
                                    "usage limit", "429", "resets"))
                if rate_limited:
                    sleep = min(900, 60 * (attempt + 1))
                    print(f"    [rate-limited] {e!s:.120s}  sleeping {sleep}s "
                          f"(attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep)
                    continue
                # Non-rate-limit transient: short exponential backoff.
                time.sleep(min(60, 2 ** attempt))
        # Out of retries -> raise a marker exception so the outer cache layer
        # can re-raise it (joblib must NOT cache this failure).
        raise ClaudeRetriesExhausted(
            f"Claude call failed after {max_retries} retries: {last_err}"
        )


class ClaudeRetriesExhausted(RuntimeError):
    """Raised when ClaudeHaiku runs out of retries (rate limit, empty resp)."""


# ---------------------------------------------------------------------------
# Slightly perturbed prompt.
# ---------------------------------------------------------------------------
# Original (in evolve/src/interp_eval.py:ask_llm):
#   "Here is a trained regression model:\n\n{model_str}\n\n{question}"
# Perturbed: re-worded preamble + explicit instruction to rely only on the
# printed model. Same content, different phrasing.

_ORIGINAL_ASK_LLM = interp_eval.ask_llm

def perturbed_ask_llm(llm, model_str, question, max_tokens=200):
    prompt = (
        "Below is a fitted regression model rendered as a string.\n"
        "Read the model carefully and answer the following question, "
        "using ONLY the information present in the printed model.\n\n"
        f"---\nMODEL:\n{model_str}\n---\n\n"
        f"QUESTION: {question}"
    )
    return llm(
        prompt,
        max_completion_tokens=max_tokens,
        stop=["cannot", "I do not have enough", "I'm sorry"],
    )

interp_eval.ask_llm = perturbed_ask_llm


# ---------------------------------------------------------------------------
# Cached per-test runner.
# ---------------------------------------------------------------------------

_cache = Memory(location=str(RESULTS_DIR / "cache"), verbose=0)


@_cache.cache
def _run_one_test(model_name, test_fn_name, model, _cache_key="claude-haiku-v1"):
    llm = ClaudeHaiku()
    test_fn = _ALL_NEW_TEST_FNS[test_fn_name]
    try:
        result = test_fn(model, llm)
    except AssertionError as e:
        result = dict(test=test_fn_name, passed=False,
                      error=f"Assertion: {e}", response=None)
    except ClaudeRetriesExhausted:
        # Re-raise so joblib does NOT cache this; we want to retry on a
        # subsequent invocation once rate limits have reset.
        raise
    except Exception as e:
        result = dict(test=test_fn_name, passed=False,
                      error=str(e), response=None)
    result["model"] = model_name
    result.setdefault("test", test_fn_name)
    return result


def run_all_interp_tests(model_defs_simple, n_jobs=8):
    tasks = [(name, reg, test_fn)
             for name, reg in model_defs_simple
             for test_fn in ALL_NEW_TESTS]

    print(f"  Running {len(tasks)} test calls with Claude Haiku "
          f"({len(model_defs_simple)} models × {len(ALL_NEW_TESTS)} tests, "
          f"n_jobs={n_jobs})")

    # Run with thread parallelism. Tasks that raise ClaudeRetriesExhausted
    # propagate up (joblib re-raises), so we wrap each call to swallow them
    # and continue: their cache is left empty so a subsequent invocation
    # retries them after rate limits reset.
    def _safe_run(name, reg, test_fn):
        try:
            return _run_one_test(name, test_fn.__name__, reg)
        except ClaudeRetriesExhausted as e:
            return {"_skipped": True, "model": name,
                    "test": test_fn.__name__, "error": str(e)[:200]}

    raw = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_safe_run)(name, reg, test_fn)
        for name, reg, test_fn in tasks
    )
    results = [r for r in raw if not r.get("_skipped")]
    skipped = [r for r in raw if r.get("_skipped")]
    print(f"\n  Final: {len(results)}/{len(tasks)} succeeded, "
          f"{len(skipped)} skipped (re-run to retry).")

    # Per-model summary print.
    for name, _ in model_defs_simple:
        sub = [r for r in results if r["model"] == name]
        passed = sum(r.get("passed", False) for r in sub)
        print(f"    {name:35s}  {passed:3d} / {len(sub)} succeeded")
    return results


# ---------------------------------------------------------------------------
# CSV writers (matching the GPT-5.4 results layout).
# ---------------------------------------------------------------------------

def write_per_test_csv(results, path):
    fieldnames = ["model", "test", "passed", "ground_truth", "response", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Wrote {path}")


def write_overall_csv(results, model_defs_simple, perf_rank, path):
    fieldnames = ["commit", "mean_rank", "frac_interpretability_tests_passed",
                  "status", "model_name", "description"]
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, _ in model_defs_simple:
            sub = by_model.get(name, [])
            frac = (sum(r.get("passed", False) for r in sub) / len(sub)
                    if sub else float("nan"))
            w.writerow(dict(
                commit="",
                mean_rank=perf_rank.get(name, ""),
                frac_interpretability_tests_passed=f"{frac:.4f}",
                status="baseline" if name in {n for n, _ in BASELINE_DEFS} else "evolved",
                model_name=name,
                description=BASELINE_DESCRIPTIONS.get(name, ""),
            ))
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(smoke_test=False):
    print("=" * 70)
    print("CLAUDE SONNET REPLAY of new_results_gpt_5_4 (perturbed prompt)")
    print("=" * 70)
    print(f"  Output dir: {RESULTS_DIR}")
    print(f"  Model:      {CLAUDE_MODEL}")

    # Load models (16 baselines + 25 evolved from apr9).
    print("\nLoading models from apr9 evolved library + baselines ...")
    evolved = load_all_models()  # [(name, model_instance, description), ...]
    print(f"  {len(evolved)} evolved models loaded")
    global ALL_NEW_TESTS
    if smoke_test:
        bdefs = [BASELINE_DEFS[0]]   # PyGAM
        edefs_3 = evolved[:1]
        ALL_NEW_TESTS = ALL_NEW_TESTS[:3]  # type: ignore[assignment]
    else:
        bdefs = BASELINE_DEFS
        edefs_3 = evolved
        print(f"  Using all {len(ALL_NEW_TESTS)} tests "
              f"(matches GPT-5.4 evaluation scope, 41 models × {len(ALL_NEW_TESTS)} tests)")

    # Normalize to (name, model) pairs.
    model_defs_simple = [(n, reg) for n, reg in bdefs] + \
                        [(n, reg) for n, reg, _ in edefs_3]

    # Run tests.
    print("\nRunning interpretability tests ...")
    results = run_all_interp_tests(model_defs_simple, n_jobs=8 if not smoke_test else 1)

    # Write per-test + overall CSVs (mean_rank pulled from the GPT-5.4 sister
    # CSV so the y-axis is identical between figures).
    perf_rank = {}
    sister_csv = SCRIPT_DIR / "new_results_gpt_5_4" / "overall_results_gpt54.csv"
    if sister_csv.exists():
        with open(sister_csv) as f:
            for r in csv.DictReader(f):
                if r["mean_rank"] not in ("", "nan"):
                    perf_rank[r["model_name"]] = r["mean_rank"]

    write_per_test_csv(results, RESULTS_DIR / "interpretability_results_claude_haiku.csv")
    write_overall_csv(results, model_defs_simple, perf_rank,
                      RESULTS_DIR / "overall_results_claude_haiku.csv")
    print("\nDone.")


if __name__ == "__main__":
    main(smoke_test="--smoke" in sys.argv)
