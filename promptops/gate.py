"""The decision layer -- the actual product.

Two entry points:
  gate_new_version()   -> called when someone edits a prompt directly
  handle_template_edit() -> called when a shared template changes (blast radius)

Both share the three-tier rule:
  new >= previous            -> GREEN, do nothing
  threshold <= new < previous-> AMBER, flag for manual rollback (human decides)
  new < threshold            -> RED, auto rollback to last known-good
"""
from dataclasses import dataclass

from . import config, db, evaluator, graph, llm


@dataclass
class Decision:
    prompt_id: str
    new_score: float
    prev_score: float | None
    status: str
    action: str           # "none" | "flag_manual" | "auto_rollback"
    explanation: str = ""  # populated (in an Indian language) on regressions


def _classify(new_score: float, prev_score: float | None) -> tuple[str, str]:
    if prev_score is None or new_score >= prev_score:
        return config.GREEN, "none"
    if new_score >= config.THRESHOLD:
        return config.AMBER, "flag_manual"
    return config.RED, "auto_rollback"


# --------------------------------------------------------------------------
# Case 1: someone edited a prompt directly (its own version history)
# --------------------------------------------------------------------------
def gate_new_version(prompt_id: str, version: int) -> Decision:
    row = db.get_prompt_version(prompt_id, version)
    new_score = evaluator.evaluate(row)

    prev_score = None
    if version > 1:
        prev = db.get_prompt_version(prompt_id, version - 1)
        prev_score = prev["score"] if prev else None

    status, action = _classify(new_score, prev_score)
    db.set_prompt_result(prompt_id, version, new_score, status)
    graph.set_status(prompt_id, new_score, status)

    dec = Decision(prompt_id, new_score, prev_score, status, action)
    if status == config.RED:
        # roll the prompt's own body back to the previous version by creating a
        # fresh version that reuses the last-good body (kept explicit for audit).
        prev = db.get_prompt_version(prompt_id, version - 1)
        db.add_prompt_version(
            prompt_id, prev["body"], prev["dataset"],
            template_id=prev["template_id"],
            pinned_template_version=prev["pinned_template_version"],
        )
    return dec


# --------------------------------------------------------------------------
# Case 2: a shared template changed -> re-eval every dependent (blast radius)
# --------------------------------------------------------------------------
def handle_template_edit(template_id: str, new_text: str) -> list[Decision]:
    new_version = db.add_template_version(template_id, new_text)
    graph.upsert_template(template_id, new_version)

    decisions: list[Decision] = []
    for p in db.prompts_using_template(template_id):
        pid, pver = p["id"], p["version"]
        prev_score = p["score"]  # score under the template version it's pinned to

        # Score the prompt AS IF it adopted the new template version (Option B:
        # this is what it would actually run once the pin moves forward).
        new_score = evaluator.evaluate(p, template_version=new_version)
        status, action = _classify(new_score, prev_score)

        if status == config.RED:
            # Surgical rollback: pin THIS prompt back to its last-good template
            # version. Others that survived stay on the new one. That contrast
            # -- some nodes snap back, some don't -- is the demo's money shot.
            good = db.last_passing_template_version(pid, template_id)
            db.set_pin(pid, pver, good)
            graph.set_pin(pid, template_id, good)
            recovered = evaluator.evaluate(db.get_prompt_version(pid, pver))
            db.set_prompt_result(pid, pver, recovered, config.GREEN)
            graph.set_status(pid, recovered, config.GREEN)
            explanation = llm.explain_regression(
                pid, template_id, prev_score or 0.0, new_score
            )
            decisions.append(
                Decision(pid, new_score, prev_score, config.RED,
                         "auto_rollback", explanation)
            )
        else:
            # green or amber: adopt the new template version, record status
            db.set_pin(pid, pver, new_version)
            graph.set_pin(pid, template_id, new_version)
            db.set_prompt_result(pid, pver, new_score, status)
            graph.set_status(pid, new_score, status)
            explanation = ""
            if status == config.AMBER:
                explanation = llm.explain_regression(
                    pid, template_id, prev_score or 0.0, new_score
                )
            decisions.append(
                Decision(pid, new_score, prev_score, status,
                         "flag_manual" if status == config.AMBER else "none",
                         explanation)
            )
    return decisions
