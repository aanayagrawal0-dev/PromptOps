"""The pre-seeded, reproducible-as-a-real-model-allows demo scenario.

  Template  compliance_disclaimer v1  -> requires a financial-advice disclosure
  Prompts   five, all wired to it, all green at v1.

Then in the demo you POST the template edit that drops the disclosure. Because
each prompt's eval checks lean on the shared disclaimer differently, the one
edit produces two distinct outcomes:

  stock_advisor / fund_summary / sip_explainer / market_news  -> all four
      require the disclosure concept -> losing it drops their real score
      below threshold -> RED -> auto-rollback to v1 -> back to green, with a
      Hindi explanation attached.
  greeting_bot -> its check ("a warm, friendly greeting") never depended on
      the disclosure -> the edit shouldn't move its score -> stays GREEN.

Checks are real concepts, judged semantically by a live model
(llm.judge_presence) -- not literal strings -- so exact wording/fractions
aren't guaranteed to reproduce identically run to run the way the old
deterministic mock did. An AMBER ("flagged, partial regression") outcome was
part of the original mock-only design but was dropped here: it depended on
several filler checks landing at an exact fraction, which is fragile once
scoring runs through a real, non-deterministic judge instead of a script.
"""
from . import db, graph

DISCLAIMER_V1 = (
    "Keep your entire answer to at most 3 sentences -- no headers, no bullet "
    "points, no numbered lists, no bull-case/bear-case breakdowns. Somewhere "
    "in those 3 sentences, include a short note that this is not financial "
    "advice and not a recommendation to buy or sell."
)

# A ready-made breaking edit (used live in the demo, not seeded): it drops the
# disclosure instruction. Note it deliberately avoids the word "concise" so the
# mock model keeps its normal wording -- that's what lets market_news land on
# AMBER (only the disclosure check fails) instead of crashing to RED.
DISCLAIMER_V2_BAD = "Respond in a friendly, professional tone."

# Core prompts: every eval row requires the disclosure, so losing it is fatal.
# Checks are real concepts judged semantically in live mode (see
# llm.judge_presence), and literally substring-matched in mock mode -- kept
# short/plain so the same phrase works as an honest concept AND as something
# the offline mock's canned wording can actually contain verbatim.
_DATASET = [
    {"input": "Should I buy TCS stock?",
     "checks": {"must_contain": ["not financial advice"]}},
    {"input": "Is this a good time to invest in an index fund?",
     "checks": {"must_contain": ["not financial advice"]}},
]

# market_news depends on the same disclosure -- same shape as the dataset
# above, deliberately kept to ONE real check. (Previously this had 4 extra
# filler phrases tuned to land at exactly 4/5 = 0.80 against the old mock's
# fixed wording -- with a real, non-deterministic judge, chasing an exact
# fraction across several checks is fragile rather than illustrative.)
_MARKET_NEWS_DATASET = [
    {"input": "Summarize today's market movement.",
     "checks": {"must_contain": ["not financial advice"]}},
]

# greeting_bot: its check is unrelated to the disclosure, so editing
# compliance_disclaimer shouldn't move its score -> stays GREEN. (mock mode's
# canned reply for a greeting-style input now actually contains a greeting --
# see llm._mock_run -- so this check is honest in both modes, not a leftover
# artifact of unrelated mock wording like the old "regarding" check was.)
_GREETING_DATASET = [
    {"input": "Greet the user warmly.",
     "checks": {"must_contain": ["greeting"]}},
]

# id -> (body, dataset). All reference the shared {compliance_disclaimer}.
_PROMPTS = {
    "stock_advisor": _DATASET,
    "fund_summary": _DATASET,
    "sip_explainer": _DATASET,
    "market_news": _MARKET_NEWS_DATASET,
    "greeting_bot": _GREETING_DATASET,
}

# 3rd dependency tier, above prompts: real downstream consumers (an app or
# agent) that call one or more of the prompts above. A pipeline has no score
# of its own -- see main.py's _pipeline_status, which computes its badge live
# from whatever its prompts' current status actually is. greeting_bot is
# deliberately shared by both, so a regression in one prompt can be shown
# rippling into more than one real downstream consumer at once.
_PIPELINES = {
    "advisor_app": ("Customer Advisor App", ["stock_advisor", "sip_explainer", "greeting_bot"]),
    "market_digest": ("Daily Market Digest Bot", ["market_news", "fund_summary", "greeting_bot"]),
}


def seed():
    db.init_db()

    tv = db.add_template_version("compliance_disclaimer", DISCLAIMER_V1)
    graph.upsert_template("compliance_disclaimer", tv)

    # Create AND score each prompt in the same pass, against the version that
    # was actually just created -- not a hardcoded "1". Re-running seed() on
    # an already-seeded DB creates version 2, 3, ... each time, and each of
    # those needs its own score, not a copy of whatever v1 got.
    from . import evaluator, gate
    for pid, dataset in _PROMPTS.items():
        body = "You are a helpful assistant. {compliance_disclaimer}"
        pv = db.add_prompt_version(
            pid, body, dataset,
            template_id="compliance_disclaimer",
            pinned_template_version=tv,
        )
        graph.upsert_prompt(pid, pv, "compliance_disclaimer", tv)

        row = db.get_prompt_version(pid, pv)
        score = evaluator.evaluate(row)
        status = gate.score_band(score)
        db.set_prompt_result(pid, pv, score, status)
        graph.set_status(pid, score, status)

    for pl_id, (name, prompt_ids) in _PIPELINES.items():
        db.add_pipeline(pl_id, name, prompt_ids)

    print(f"Seeded: 1 template, {len(_PROMPTS)} dependent prompts, "
          f"{len(_PIPELINES)} downstream pipelines, all green.")


if __name__ == "__main__":
    seed()
