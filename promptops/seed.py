"""The pre-seeded, reproducible demo scenario.

  Template  compliance_disclaimer v1  -> requires the 'not financial advice' line
  Prompts   stock_advisor / fund_summary / sip_explainer  all depend on it
  All three green at v1.

Then in the demo you POST the template edit that drops the disclosure. Watch
the graph: all three go red, auto-rollback pins them back to v1, they go green
again, and Sarvam prints why -- in Hindi.
"""
from . import db, graph

DISCLAIMER_V1 = (
    "Always end your answer by telling the user this is not financial advice "
    "and not a recommendation to buy or sell."
)

# The edit that breaks everything downstream (used in the demo, not seeded):
DISCLAIMER_V2_BAD = "Keep every response concise."

_DATASET = [
    {"input": "Should I buy TCS stock?",
     "checks": {"must_contain": ["not financial advice"]}},
    {"input": "Is this a good time to invest in an index fund?",
     "checks": {"must_contain": ["not financial advice"]}},
]


def seed():
    db.init_db()
    db.reset_demo()

    tv = db.add_template_version("compliance_disclaimer", DISCLAIMER_V1)
    graph.upsert_template("compliance_disclaimer", tv)

    for pid in ("stock_advisor", "fund_summary", "sip_explainer"):
        body = f"You are a helpful investing assistant. {{compliance_disclaimer}}"
        pv = db.add_prompt_version(
            pid, body, _DATASET,
            template_id="compliance_disclaimer",
            pinned_template_version=tv,
        )
        graph.upsert_prompt(pid, pv, "compliance_disclaimer", tv)

    # Score everyone at v1 so they start green (baseline the demo rolls back to).
    from . import evaluator, config
    for pid in ("stock_advisor", "fund_summary", "sip_explainer"):
        row = db.get_prompt_version(pid, 1)
        score = evaluator.evaluate(row)
        db.set_prompt_result(pid, 1, score, config.GREEN)
        graph.set_status(pid, score, config.GREEN)

    print("Seeded: 1 template, 3 dependent prompts, all green.")


if __name__ == "__main__":
    seed()
