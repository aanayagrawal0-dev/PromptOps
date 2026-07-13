"""FastAPI surface. Only the endpoints the demo actually needs -- CRUD for
prompts/templates is left as an exercise (standard, uninteresting)."""
import os
from datetime import datetime, timedelta

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, evaluator, gate, graph, llm, resolver, seed as seed_module

app = FastAPI(title="PromptOps core")

# The frontend is served from a different origin (e.g. http://localhost:5500)
# than the API (http://localhost:8000), so the browser needs CORS to allow the
# fetch() calls. Wide-open is fine for a local dev/demo backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(requests.exceptions.RequestException)
def _sarvam_call_failed(request, exc: requests.exceptions.RequestException):
    """Any endpoint that ends up calling the live Sarvam API (llm.py) can hit a
    real upstream failure -- bad/expired key, out-of-credit account, rate
    limit surviving both retries, timeout, DNS/network hiccup. Without this,
    that exception was unhandled and fell through to FastAPI's generic 500,
    which the frontend then reported as just 'POST /x -> 500' -- true, but
    useless for telling a network blip apart from an auth failure apart from
    a dead API key. Surface Sarvam's own error body when it sent one, so the
    dashboard can show the real reason instead of a mystery crash."""
    detail = str(exc)
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = resp.json()
            detail = body.get("error", {}).get("message") or body.get("message") or resp.text
        except Exception:
            detail = resp.text or detail
    print(f"[main] Sarvam call failed on {request.method} {request.url.path}: {detail}")
    return JSONResponse(status_code=502, content={"detail": f"Sarvam API call failed: {detail}"})


class TemplateEdit(BaseModel):
    template_id: str
    new_text: str


class PromptEdit(BaseModel):
    prompt_id: str
    version: int


class PromptBodyEdit(BaseModel):
    body: str


class NewTemplate(BaseModel):
    template_id: str
    text: str = ""


class NewPrompt(BaseModel):
    prompt_id: str
    body: str = ""
    template_id: str | None = None
    # Optional eval checks. Each phrase becomes a `must_contain` rule the
    # resolved prompt's output is scored against.
    must_contain: list[str] = []
    # What input this prompt is actually meant to handle. Without this, there
    # is no honest way to test it -- a check can only mean something if it's
    # judged against an output the prompt would realistically produce.
    sample_input: str = ""


class NewPipeline(BaseModel):
    pipeline_id: str
    name: str = ""
    # Which tracked prompts this pipeline actually calls. A pipeline with no
    # prompts is a config mistake, not a valid registration -- caught below.
    prompt_ids: list[str] = []


@app.on_event("startup")
def _startup():
    db.init_db()


@app.post("/demo/seed")
def demo_seed():
    seed_module.seed()
    return {"ok": True, "graph": graph.snapshot()}


@app.post("/demo/reset")
def demo_reset():
    """What 'Reset Graph' actually does now: wipe every tracked prompt and
    template (including everything registered during testing) and reseed
    the original 5-prompt baseline -- not just reload the page, which left
    all that test data sitting in the database untouched."""
    db.reset_all()
    seed_module.seed()
    return {"ok": True, "graph": graph.snapshot()}


@app.post("/prompts/{prompt_id}/edit")
def edit_prompt_body(prompt_id: str, body: PromptBodyEdit):
    """Edit a prompt's own body directly (not a shared template) and
    re-evaluate it for real against its existing checks -- this is what a
    node's edit box in the dependency graph calls. If the new body breaks
    the checks, gate.gate_new_version auto-rolls it back to the last-good
    body: the same auto-heal principle a template edit gets, scoped to one
    prompt's own content instead of a blast radius."""
    current_version = db.latest_prompt_version(prompt_id)
    if current_version is None:
        raise HTTPException(status_code=404, detail=f"no such prompt: {prompt_id}")
    current = db.get_prompt_version(prompt_id, current_version)

    new_version = db.add_prompt_version(
        prompt_id, body.body, current["dataset"],
        template_id=current["template_id"],
        pinned_template_version=current["pinned_template_version"],
    )
    graph.upsert_prompt(prompt_id, new_version, current["template_id"], current["pinned_template_version"])

    # Show what the model actually said for one real test input, so a score
    # you don't trust can be checked against real output instead of staying
    # a mystery -- same principle as the registration flow. Only the first
    # dataset row (gate_new_version's own evaluate() call still covers all
    # of them for the real score); showing every row would double the live
    # calls this one edit costs.
    row = db.get_prompt_version(prompt_id, new_version)
    sample_input, model_output = None, None
    if row["dataset"]:
        sample_input = row["dataset"][0]["input"]
        model_output = llm.run_prompt(resolver.resolve(row), sample_input)

    decision = gate.gate_new_version(prompt_id, new_version)
    return {
        "id": prompt_id,
        "version": new_version,
        "score": decision.new_score,
        "status": decision.status,
        "action": decision.action,
        "rolled_back": decision.action == "auto_rollback",
        "sample_input": sample_input,
        "model_output": model_output,
        "dataset_row_count": len(row["dataset"]),
    }


@app.post("/prompts/{prompt_id}/unlink")
def unlink_prompt(prompt_id: str):
    """Remove a prompt's dependency on whatever shared template it's pinned
    to. It keeps its own body/history -- it just stops resolving against,
    and stops being affected by edits to, that template."""
    version = db.latest_prompt_version(prompt_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"no such prompt: {prompt_id}")

    db.unlink_template(prompt_id, version)
    graph.unlink(prompt_id)

    row = db.get_prompt_version(prompt_id, version)
    score = evaluator.evaluate(row)
    status = gate.score_band(score)
    db.set_prompt_result(prompt_id, version, score, status)
    graph.set_status(prompt_id, score, status)

    return {
        "id": prompt_id, "version": version, "template_id": None,
        "score": score, "status": status,
    }


@app.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: str):
    """Permanently remove a prompt (every version). For cleaning up a single
    junk registration -- /demo/reset is the nuclear option that wipes and
    reseeds everything; this only touches the one prompt you asked for."""
    version = db.latest_prompt_version(prompt_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"no such prompt: {prompt_id}")
    db.delete_prompt(prompt_id)
    graph.delete(prompt_id)
    return {"ok": True, "id": prompt_id}


@app.get("/regressions")
def list_regressions():
    """Real regression history -- every auto-rollback or flagged-for-review
    event the gate has actually logged (see gate.handle_template_edit),
    including a real measured 'Avg. Recovery Time', not a placeholder
    number. Powers the Regressions page."""
    events = db.list_regression_events(limit=50)
    cutoff = datetime.utcnow() - timedelta(days=7)

    detected_7d = 0
    auto_healed = 0
    flagged = 0
    durations = []
    for e in events:
        try:
            ts = datetime.strptime(e["created_at"], "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            ts = None
        if ts and ts >= cutoff:
            detected_7d += 1
        if e["action"] == "auto_rollback":
            auto_healed += 1
        elif e["action"] == "flag_manual":
            flagged += 1
        if e["duration_ms"] is not None:
            durations.append(e["duration_ms"])

    return {
        "events": events,
        "stats": {
            "detected_7d": detected_7d,
            "auto_healed": auto_healed,
            "flagged": flagged,
            "avg_recovery_ms": round(sum(durations) / len(durations)) if durations else None,
        },
    }


@app.get("/prompts")
def list_prompts():
    """Every tracked prompt with its live score/status, straight from SQLite.
    This is what the console + dependency views load on startup."""
    prompts = [
        {
            "id": p["id"],
            "version": p["version"],
            "score": p["score"],
            "status": p["status"],
            "template_id": p["template_id"],
            "pinned_template_version": p["pinned_template_version"],
            # Body + eval checks so the UI can show what a prompt actually is
            # on click, not just its score -- already loaded off the same
            # row, so no extra query per prompt.
            "body": p["body"],
            "dataset": p["dataset"],
        }
        for p in db.all_prompts()
    ]
    return {"threshold": config.THRESHOLD, "prompts": prompts}


@app.get("/templates/{template_id}")
def get_template(template_id: str):
    """Current text of a template, so the dashboard can load the real thing
    into its editor instead of a hardcoded string."""
    t = db.get_latest_template(template_id)
    if not t:
        return {"template_id": template_id, "version": None, "text": ""}
    return {"template_id": t["id"], "version": t["version"], "text": t["text"]}


@app.get("/graph")
def get_graph():
    """Full dependency graph for the blast-radius view."""
    return graph.snapshot()


@app.get("/graph/blast-radius/{template_id}")
def get_blast_radius(template_id: str):
    return {"template_id": template_id, "dependents": graph.blast_radius(template_id)}


@app.post("/templates/edit")
def edit_template(body: TemplateEdit):
    """THE demo endpoint. Edit a shared template -> re-eval every dependent ->
    tiered gate -> auto-rollback the red ones -> Sarvam explains why."""
    decisions = gate.handle_template_edit(body.template_id, body.new_text)
    return {
        "template_id": body.template_id,
        "decisions": [d.__dict__ for d in decisions],
        "graph": graph.snapshot(),
    }


@app.post("/prompts/gate")
def gate_prompt(body: PromptEdit):
    """Called after a direct prompt edit -> score + tiered gate on its own history."""
    return gate.gate_new_version(body.prompt_id, body.version).__dict__


@app.post("/templates")
def create_template(body: NewTemplate):
    """Register a brand-new shared template (v1). Prompts can then depend on it."""
    version = db.add_template_version(body.template_id, body.text)
    graph.upsert_template(body.template_id, version)
    return {"template_id": body.template_id, "version": version, "text": body.text}


_GENERIC_BASELINE_PROMPT = "You are a helpful assistant."


@app.post("/prompts")
def create_prompt(body: NewPrompt):
    """Register a new tracked prompt. It's pinned to the current version of the
    template it depends on, scored against its (optional) eval checks against a
    REAL sample input (not a placeholder), and shows up in the console + blast
    radius from then on."""
    sample_input = body.sample_input.strip()
    dataset = (
        [{"input": sample_input or "Tell me about yourself.",
          "checks": {"must_contain": body.must_contain}}]
        if body.must_contain else []
    )
    tv = db.latest_template_version(body.template_id) if body.template_id else None
    version = db.add_prompt_version(
        body.prompt_id, body.body, dataset,
        template_id=body.template_id, pinned_template_version=tv,
    )
    graph.upsert_prompt(body.prompt_id, version, body.template_id, tv)

    row = db.get_prompt_version(body.prompt_id, version)
    warnings: list[str] = []

    # Catch a real, easy-to-make mistake at registration time: declaring a
    # dependency on a shared template but never actually referencing it in
    # the body means that dependency is inert -- the template's text is
    # never spliced in, so editing that template will never affect this
    # prompt, even though the UI will show it as a dependent.
    if body.template_id and f"{{{body.template_id}}}" not in body.body:
        warnings.append(
            f"This depends on '{body.template_id}' but never references "
            f"{{{body.template_id}}} in its body, so that template's text is "
            f"never actually spliced in. Editing '{body.template_id}' won't "
            f"affect this prompt, despite the dependency link shown."
        )

    # Run + score against the real sample input, and keep the raw model
    # output so the caller can see WHY it passed or failed, not just trust
    # a badge. (evaluator.evaluate() re-does this same resolve+run+score
    # internally; done by hand here since we also want the output back.)
    model_output = None
    if dataset:
        system_prompt = resolver.resolve(row)
        model_output = llm.run_prompt(system_prompt, dataset[0]["input"])
        score = evaluator.score_output(model_output, dataset[0]["checks"])

        # A check that also passes with NO real instructions isn't testing
        # anything this prompt is actually responsible for -- it's testing
        # the model's default behavior. A vague check like "helpful" or
        # "Help people" is satisfied by almost any competent reply, prompt
        # or no prompt, so a green badge from it doesn't mean the prompt
        # itself does its job. Only worth the extra live call when the real
        # score already looks healthy -- that's the case where this false
        # confidence actually matters.
        if score >= config.HEALTHY_THRESHOLD:
            baseline_output = llm.run_prompt(_GENERIC_BASELINE_PROMPT, dataset[0]["input"])
            baseline_score = evaluator.score_output(baseline_output, dataset[0]["checks"])
            if baseline_score >= config.HEALTHY_THRESHOLD:
                warnings.append(
                    "This check also passes with a completely generic system "
                    "prompt -- no instructions from this prompt at all. It may "
                    "be testing the model's default behavior, not anything "
                    "this prompt is actually responsible for causing. Try a "
                    "more specific check (an exact requirement your prompt's "
                    "instructions introduce, not a general quality like "
                    "'helpful' or 'polite')."
                )
    else:
        score = 1.0

    # A fresh registration has no prior version to regress against, so its
    # baseline status is just the absolute score band (same bands the
    # dashboard Legend and gate.py use -- no separate GREEN/RED-only rule).
    status = gate.score_band(score)
    db.set_prompt_result(body.prompt_id, version, score, status)
    graph.set_status(body.prompt_id, score, status)

    return {
        "id": body.prompt_id,
        "version": version,
        "score": score,
        "status": status,
        "template_id": body.template_id,
        "sample_input": dataset[0]["input"] if dataset else None,
        "model_output": model_output,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Pipelines -- the 3rd dependency tier: real downstream consumers (an app or
# agent) that call one or more tracked prompts. A pipeline's status is never
# stored -- it's computed here, live, every time it's asked for, as the worst
# status among the prompts it actually depends on (red beats amber beats
# green). That means a pipeline node can never show a badge that's out of
# sync with what its prompts are doing right now: if a template edit auto-
# rolls a prompt back to green, every pipeline built on it is healthy again
# on the very next /pipelines call, with no separate re-eval step to forget.
# --------------------------------------------------------------------------
_STATUS_RANK = {config.GREEN: 0, config.AMBER: 1, config.RED: 2}


def _pipeline_status(prompt_ids: list[str], prompt_lookup: dict) -> dict:
    worst_status = config.GREEN
    worst_prompt = None
    scores = []
    missing = []
    for pid in prompt_ids:
        p = prompt_lookup.get(pid)
        if p is None:
            missing.append(pid)
            continue
        scores.append(p["score"])
        status = p["status"] or config.GREEN
        if _STATUS_RANK.get(status, 0) > _STATUS_RANK.get(worst_status, 0):
            worst_status = status
            worst_prompt = pid
    return {
        "status": worst_status,
        "worst_prompt": worst_prompt,
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        # prompt_ids the pipeline still references that no longer exist
        # (e.g. deleted from the console) -- surfaced instead of silently
        # dropped, since that's a real config problem for the pipeline.
        "missing_prompt_ids": missing,
    }


@app.get("/pipelines")
def list_pipelines():
    """Every registered pipeline with its live-computed status. Powers the
    3rd row of the dependency graph."""
    prompt_lookup = {p["id"]: p for p in db.all_prompts()}
    pipelines = db.all_pipelines()
    out = []
    for pl in pipelines:
        status_info = _pipeline_status(pl["prompt_ids"], prompt_lookup)
        out.append({**pl, **status_info})
    return {"pipelines": out}


@app.post("/pipelines")
def create_pipeline(body: NewPipeline):
    """Register a downstream consumer and the prompts it actually calls.
    Re-registering the same pipeline_id replaces its prompt links, so this
    also doubles as an edit."""
    if not body.prompt_ids:
        raise HTTPException(
            status_code=400,
            detail="a pipeline needs at least one prompt_id -- a pipeline "
                   "that depends on nothing isn't a real dependency link.",
        )
    unknown = [pid for pid in body.prompt_ids if db.latest_prompt_version(pid) is None]
    if unknown:
        raise HTTPException(
            status_code=404,
            detail=f"no such prompt(s): {', '.join(unknown)}",
        )
    db.add_pipeline(body.pipeline_id, body.name or body.pipeline_id, body.prompt_ids)
    prompt_lookup = {p["id"]: p for p in db.all_prompts()}
    status_info = _pipeline_status(body.prompt_ids, prompt_lookup)
    return {
        "id": body.pipeline_id,
        "name": body.name or body.pipeline_id,
        "prompt_ids": body.prompt_ids,
        **status_info,
    }


@app.delete("/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id: str):
    pipelines = {p["id"] for p in db.all_pipelines()}
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail=f"no such pipeline: {pipeline_id}")
    db.delete_pipeline(pipeline_id)
    return {"ok": True, "id": pipeline_id}


@app.get("/")
def _root():
    return RedirectResponse(url="/landing.html")


# Serve the static frontend from this same FastAPI app, at the same origin as
# the API -- one deploy, one URL, no separate static host, no CORS to get
# wrong. Deliberately NOT a blanket mount of the whole project directory:
# that would also serve .env (the Sarvam API key), promptops.db, and the
# Python source itself over plain HTTP to anyone who requested them.
# Only assets/ (css/js, nothing sensitive) and an explicit allowlist of the
# actual page files are exposed.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/assets", StaticFiles(directory=os.path.join(_PROJECT_ROOT, "assets")), name="assets")

_PAGES = [
    "landing.html", "dashboard.html", "console.html",
    "new-template.html", "onboarding.html", "regressions.html",
]
for _page in _PAGES:
    def _make_handler(filename: str):
        def _serve():
            return FileResponse(os.path.join(_PROJECT_ROOT, filename))
        return _serve
    app.get(f"/{_page}")(_make_handler(_page))
