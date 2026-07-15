# PromptOps

**Catches when editing a shared prompt template silently breaks other prompts, auto-rolls back the broken ones, and explains regressions in Hindi via Sarvam AI.**


---

## The problem

Prompts in production often share common building blocks -- a compliance disclaimer, a formatting rule, a safety instruction -- pulled in by reference at runtime. Edit that shared piece once, and every prompt that depends on it changes behavior immediately, with no one having touched those prompts directly. Nothing tells you what just broke until a user notices.

PromptOps closes that gap: every prompt (and every downstream app/agent that calls those prompts) that depends on an edited template is automatically re-evaluated the moment the template changes, before the change reaches anyone.

## How it works

1. **A shared template changes.** Templates are resolved *live* -- a prompt doesn't store a template's text, it stores a reference to it (`{compliance_disclaimer}`), so editing the template changes what every dependent prompt actually runs, immediately.
2. **Every dependent is re-scored against a real model call.** Not the whole prompt library -- only prompts with a real dependency edge into the specific template that changed. Each is scored against its own eval checks. In live mode, each check is a *concept* ("a disclosure that this isn't financial advice"), judged semantically by a real Sarvam model call that allows paraphrasing -- not a literal string match. A deterministic offline mock mode (literal substring matching) exists for demos with no network/API key.
3. **A three-tier gate decides what happens next:**
   - **Score improves or holds** -> adopts the new template version, no action needed.
   - **Score dips but stays above threshold (0.8)** -> flagged for manual review, left as-is.
   - **Score drops below threshold** -> automatically pinned back to the last template version that passed, and Sarvam AI generates a plain-language explanation of what broke and why -- in Hindi, verified against the actual Devanagari script in the response (with a forced retry if the model answers in English instead).
4. **The blast radius doesn't stop at prompts.** A third tier -- real downstream consumers (pipelines/agents that call one or more prompts) -- shows which actual production surfaces are affected by a prompt regression, not just the prompt itself. A pipeline's health is never stored: it's computed live, every time, as the worst status among the prompts it depends on, so it can never show a stale badge.

## Architecture

```
  Frontend                FastAPI backend            Sarvam AI
  (static HTML/JS,   -->  (promptops/*.py)      -->  (chat completions +
  same-origin,             |                          semantic judging +
  served by FastAPI)       |                          Hindi explanations)
                           v
                     SQLite (source of truth:
                     prompt/template versions,
                     scores, regression history)
                           |
                           v
                     Neo4j (dependency graph
                     mirror -- optional, off
                     by default)
```

Neo4j is optional and off by default (`PROMPTOPS_GRAPH_DISABLED=true`) -- every graph write becomes a no-op and the backend degrades gracefully. SQLite is the actual source of truth the UI reads from; Neo4j exists as a relationship/display-state mirror for anyone who wants to wire it up.

## Dependency graph -- three real tiers

```
              compliance_disclaimer (template)
             /      |       |       |       \
   stock_advisor  fund_summary  sip_explainer  market_news  greeting_bot   <- prompts
        \              /                              \        /
         advisor_app                              market_digest             <- pipelines
    (Customer Advisor App)                    (Daily Market Digest Bot)
```

Every tier is data-driven, not hardcoded -- registering a new prompt or pipeline makes it appear in the graph immediately, and the SVG canvas grows (and scrolls) to fit however many nodes actually exist instead of clipping them.

## Tech stack

- **Backend:** FastAPI, Python
- **Storage:** SQLite (dev, zero-setup; schema is standard SQL, so swapping to Postgres is a connection-string change), Neo4j for an optional dependency-graph mirror
- **LLM:** Sarvam AI (`sarvam-30b`) for semantic eval judging, output generation, and Hindi regression explanations; a deterministic offline mock mode for reproducible demos with no API key
- **Frontend:** Static HTML/CSS/JS, Tailwind (CDN), vanilla SVG for the dependency graph -- no build step, served by FastAPI itself at the same origin as the API (no separate static server, no CORS to configure for a deployed instance)

## Project structure

```
PromptOps/
|-- requirements.txt
|-- .env                        # not committed -- see Setup below
|-- landing.html
|-- onboarding.html             # static config-wizard placeholder (not yet wired to the backend)
|-- console.html                # full prompt list / health overview
|-- dashboard.html              # live 3-tier dependency graph + template edit + node inspector
|-- new-template.html           # register a new template or prompt, with real live-model testing
|-- regressions.html            # real regression history (backed by regression_events table)
|-- assets/
|   |-- api.js                  # shared fetch layer the frontend pages use
|   `-- brutalist.css
`-- promptops/
    |-- config.py                # env-driven configuration
    |-- db.py                    # SQLite layer: prompt/template versions, scores, pipelines, regression history
    |-- resolver.py               # live template resolution (the core mechanic)
    |-- evaluator.py               # scoring: resolves + runs + judges each dataset row
    |-- graph.py                    # Neo4j dependency graph mirror (soft-disables if unavailable)
    |-- gate.py                      # three-tier decision logic + auto-rollback, for both
    |                                  a direct prompt edit and a template-wide cascade
    |-- llm.py                        # Sarvam AI client (run/judge/explain) + mock mode
    |-- seed.py                        # reproducible demo scenario (1 template, 5 prompts, 2 pipelines)
    `-- main.py                        # FastAPI app / routes
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file in the project root
echo "PROMPTOPS_LLM_MODE=mock" >> .env
echo "PROMPTOPS_DB=promptops.db" >> .env
echo "PROMPTOPS_THRESHOLD=0.80" >> .env
echo "PROMPTOPS_HEALTHY_THRESHOLD=0.95" >> .env
echo "PROMPTOPS_GRAPH_DISABLED=true" >> .env

# 4. Run the backend -- this also serves the frontend, at the same origin
uvicorn promptops.main:app --reload
```

Then open `http://localhost:8000/landing.html`. The frontend seeds the demo data automatically on first load if the database is empty (no separate `curl` step needed) -- or trigger it manually with `curl -X POST http://localhost:8000/demo/seed`.

To use real Sarvam AI calls instead of the deterministic mock, set `PROMPTOPS_LLM_MODE=sarvam` and `SARVAM_API_KEY=<your key>` in `.env`. In live mode, eval checks are judged semantically by a real model call rather than literal string matching, so exact scores can vary slightly run to run -- `temperature: 0` reduces but doesn't eliminate that, which Sarvam's own docs are explicit about.

## API reference

| Method | Path | What it does |
|---|---|---|
| `POST` | `/demo/seed` | Seed the reproducible demo scenario (idempotent-ish: re-running adds new versions) |
| `POST` | `/demo/reset` | Wipe every tracked prompt/template/pipeline/regression event and reseed the baseline |
| `GET` | `/prompts` | Every tracked prompt with its live score/status/body/checks |
| `POST` | `/prompts` | Register a new prompt, scored live against a real sample input |
| `POST` | `/prompts/{id}/edit` | Edit a prompt's own body directly; re-evaluates and auto-rolls back if it breaks |
| `POST` | `/prompts/{id}/unlink` | Drop a prompt's dependency on its shared template |
| `DELETE` | `/prompts/{id}` | Permanently remove a prompt (all versions) |
| `POST` | `/prompts/gate` | Re-run the gate on an existing prompt version |
| `GET` | `/templates/{id}` | Current text of a template |
| `POST` | `/templates` | Register a new shared template |
| `POST` | `/templates/edit` | **The demo endpoint.** Edit a template, cascade-evaluate every dependent, gate + auto-rollback |
| `GET` | `/graph` | Full Neo4j dependency graph snapshot (empty if graph-disabled) |
| `GET` | `/graph/blast-radius/{id}` | Every prompt that depends on a given template |
| `GET` | `/pipelines` | Every registered pipeline, with status computed live from its prompts' current state |
| `POST` | `/pipelines` | Register a downstream consumer and which prompts it calls |
| `DELETE` | `/pipelines/{id}` | Remove a pipeline (does not touch the prompts it called) |
| `GET` | `/regressions` | Real regression history + stats (detected, auto-healed, flagged, avg recovery time) |

## Demo flow

1. `landing.html` -> `console.html` shows the seeded prompts, all healthy, all sharing the `compliance_disclaimer` template.
2. On `dashboard.html`, the graph shows all three tiers: the template at the top, its dependent prompts below it, and the two downstream pipelines (`advisor_app`, `market_digest`) below that -- `greeting_bot` deliberately feeds both, so a regression in it would ripple into two real consumers at once.
3. Load the pre-set breaking edit and apply it. Each affected prompt re-scores in sequence against a real model call; prompts that drop below threshold flag red, with a rollback available and a Hindi explanation of the regression attached. Any pipeline built on a broken prompt reflects that live, without a separate re-eval step.
4. Click any node to inspect it: a prompt's real body, its eval checks, and an editable body box that re-scores against a real live model call on save (auto-rolling back if the edit breaks it). A pipeline's node shows which prompts it calls and each one's real current status.
5. `regressions.html` shows the real, persisted history of every auto-rollback and flagged regression the gate has actually caught -- not a mockup.

## Status / known limitations

- Core gating/rollback logic, the mock-mode demo, the 3-tier dependency graph, and the live Sarvam integration are all fully functional.
- Neo4j is wired but optional and off by default -- SQLite is the real source of truth the UI reads from.
- `onboarding.html` is currently a static config-wizard mockup and isn't wired to the backend yet.
- Live mode makes real, non-deterministic model calls -- occasionally a single live call (especially the auto-rollback's own recovery verification step) can land a prompt on a borderline score purely from run-to-run model variance, not a logic bug. Re-saving a prompt's body unchanged forces a fresh live re-score.
