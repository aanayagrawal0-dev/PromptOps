"""Eval: does a real model output actually satisfy the requirements a dataset
row asks for. Score = mean pass-rate across a prompt's dataset rows.

Checks are CONCEPTS, not literal strings to grep for -- "is a compliance
disclosure present", not "does the exact substring 'not financial advice'
appear". `llm.judge_presence()` does the actual semantic call (live: a real
Sarvam judge call, allowing paraphrasing; `mock` mode: literal substring match,
so the offline demo stays 100% reproducible with no network dependency and no
API key needed). Either way `score_output` itself doesn't know or care which --
it just asks "is each concept present", then applies must/must-not polarity.

Dataset row shape:
    {
      "input": "Should I buy TCS stock?",
      "checks": {
          "must_contain":     ["a disclosure that this isn't financial advice"],
          "must_not_contain": []
      }
    }
"""
from . import llm, resolver


def score_output(output: str, checks: dict) -> float:
    must = checks.get("must_contain", [])
    must_not = checks.get("must_not_contain", [])
    all_concepts = must + must_not

    if not all_concepts:
        return 1.0

    presence = llm.judge_presence(output, all_concepts)

    passed = 0
    for i in range(len(must)):
        if presence.get(i, False):          # must be present -> pass if it is
            passed += 1
    offset = len(must)
    for j in range(len(must_not)):
        if not presence.get(offset + j, False):  # must be absent -> pass if it isn't
            passed += 1

    return passed / len(all_concepts)


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
