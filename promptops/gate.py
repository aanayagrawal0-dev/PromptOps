"""The decision layer -- the actual product.

Two entry points:
  gate_new_version()   -> called when someone edits a prompt directly
  handle_template_edit() -> called when a shared template changes (blast radius)

Status and action answer two different questions, and used to be conflated:
  status  = where the new score falls in the ABSOLUTE bands shown in the
            dashboard Legend (Healthy > 0.95, Flagged 0.8-0.95, Broken < 0.8).
            This is what the badge shows, so it can never say "HEALTHY" for a
            score the Legend itself calls flagged or broken.
  action  = what the gate actually DOES, which is regression-triggered:
      RED                      -> auto rollback to last known-good, always
                                   (it's a hard floor, regardless of trend)
      AMBER, worse than prev   -> flag for manual rollback (human decides)
      AMBER, not worse, or
      GREEN                    -> no action needed
"""
import time
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


def score_band(score: float) -> str:
    """Absolute band a score falls in -- must stay in lockstep with the
    Legend copy in dashboard.html (Healthy > 0.95, Flagged 0.8-0.95, Broken < 0.8).

    Public (not `_`-prefixed): main.py's prompt-creation endpoint uses this
    too, so a brand-new prompt gets the same three-tier badge instead of the
    old GREEN/RED-only shortcut."""
    if score < config.THRESHOLD:
        return config.RED
    if score < config.HEALTHY_THRESHOLD:
        return config.AMBER
    return config.GREEN


def _classify(new_score: float, prev_score: float | None) -> tuple[str, str]:
    status = score_band(new_score)
    regressed = prev_score is not None and new_score < prev_score

    if status == config.RED:
        action = "auto_rollback"
    elif status == config.AMBER and regressed:
        action = "flag_manual"
    else:
        action = "none"

    return status, action


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
    if action == "auto_rollback":
        # roll the prompt's own body back to the previous version by creating a
        # fresh version that reuses the last-good body (kept explicit for audit),
        # and SCORE that new version -- leaving it unscored would store
        # score=NULL/status=NULL, which the UI falls back to rendering as a
        # false "healthy" badge instead of showing what actually happened.
        prev = db.get_prompt_version(prompt_id, version - 1)
        rolled_back_version = db.add_prompt_version(
            prompt_id, prev["body"], prev["dataset"],
            template_id=prev["template_id"],
            pinned_template_version=prev["pinned_template_version"],
        )
        recovered_row = db.get_prompt_version(prompt_id, rolled_back_version)
        recovered_score = evaluator.evaluate(recovered_row)
        recovered_status = score_band(recovered_score)
        db.set_prompt_result(prompt_id, rolled_back_version, recovered_score, recovered_status)
        graph.set_status(prompt_id, recovered_score, recovered_status)
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
        t0 = time.monotonic()

        # Score the prompt AS IF it adopted the new template version (Option B:
        # this is what it would actually run once the pin moves forward).
        new_score = evaluator.evaluate(p, template_version=new_version)
        status, action = _classify(new_score, prev_score)

        if action == "auto_rollback":
            # Surgical rollback: pin THIS prompt back to its last-good template
            # version. Others that survived stay on the new one. That contrast
            # -- some nodes snap back, some don't -- is the demo's money shot.
            good = db.last_passing_template_version(pid, template_id)
            db.set_pin(pid, pver, good)
            graph.set_pin(pid, template_id, good)
            recovered = evaluator.evaluate(db.get_prompt_version(pid, pver))
            recovered_status = score_band(recovered)
            db.set_prompt_result(pid, pver, recovered, recovered_status)
            graph.set_status(pid, recovered, recovered_status)
            explanation = llm.explain_regression(
                pid, template_id, prev_score or 0.0, new_score
            )
            # Real, measured time for this prompt's re-eval + rollback --
            # not a guessed number. Powers the Regressions page's "Avg.
            # Recovery Time" stat honestly.
            db.log_regression_event(
                pid, template_id, prev_score, new_score, status,
                "auto_rollback", explanation,
                int((time.monotonic() - t0) * 1000),
                rolled_back_to_version=good,
            )
            decisions.append(
                Decision(pid, new_score, prev_score, status,
                         "auto_rollback", explanation)
            )
        else:
            # green, or amber that isn't a fresh regression: adopt the new
            # template version, record status honestly (badge always matches
            # the absolute score band, not just "did it get worse").
            db.set_pin(pid, pver, new_version)
            graph.set_pin(pid, template_id, new_version)
            db.set_prompt_result(pid, pver, new_score, status)
            graph.set_status(pid, new_score, status)
            explanation = ""
            if action == "flag_manual":
                explanation = llm.explain_regression(
                    pid, template_id, prev_score or 0.0, new_score
                )
                db.log_regression_event(
                    pid, template_id, prev_score, new_score, status,
                    "flag_manual", explanation,
                    int((time.monotonic() - t0) * 1000),
                )
            decisions.append(
                Decision(pid, new_score, prev_score, status, action, explanation)
            )
    return decisions
