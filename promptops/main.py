"""FastAPI surface. Only the endpoints the demo actually needs -- CRUD for
prompts/templates is left as an exercise (standard, uninteresting)."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import db, gate, graph, seed as seed_module

app = FastAPI(title="PromptOps core")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TemplateEdit(BaseModel):
    template_id: str
    new_text: str


class PromptEdit(BaseModel):
    prompt_id: str
    version: int


@app.on_event("startup")
def _startup():
    db.init_db()


@app.post("/demo/seed")
def demo_seed():
    seed_module.seed()
    return {"ok": True, "graph": graph.snapshot()}


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
