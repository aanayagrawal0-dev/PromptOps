"""Deterministic, rule-based eval.

Kept deliberately non-LLM so the seeded demo scenario is 100% reproducible on
stage. Each dataset row carries checks; score = mean pass-rate across rows.
Swap in an LLM-as-judge scorer later behind the same `score_output` signature.

Dataset row shape:
    {
      "input": "Should I buy TCS stock?",
      "checks": {
          "must_contain":     ["not financial advice"],
          "must_not_contain": []
      }
    }
"""
from . import llm, resolver


def score_output(output: str, checks: dict) -> float:
    out = output.lower()
    must = [s.lower() for s in checks.get("must_contain", [])]
    must_not = [s.lower() for s in checks.get("must_not_contain", [])]

    total = len(must) + len(must_not)
    if total == 0:
        return 1.0

    passed = 0
    passed += sum(1 for s in must if s in out)
    passed += sum(1 for s in must_not if s not in out)
    return passed / total


def evaluate(prompt_row: dict, template_version: int | None = None) -> float:
    """Resolve the prompt (Option B), run every dataset row, return mean score."""
    system_prompt = resolver.resolve(prompt_row, template_version=template_version)
    dataset = prompt_row["dataset"]
    if not dataset:
        return 1.0

    scores = []
    for row in dataset:
        output = llm.run_prompt(system_prompt, row["input"])
        scores.append(score_output(output, row.get("checks", {})))
    return sum(scores) / len(scores)
