"""Neo4j layer: relationships + display state for the blast-radius view.

Content and scores live in db.py; this holds the shape of the dependency graph
and each node's current colour. If GRAPH_DISABLED is set, every call is a no-op
so you can build eval/rollback logic before Neo4j is running.

Model:
    (:Prompt   {id, name, version, score, status})
    (:Template {id, name, version})
    (prompt)-[:USES_TEMPLATE {pinned_version}]->(template)
    (promptV2)-[:DERIVES_FROM]->(promptV1)
"""
from . import config

try:
    from neo4j import GraphDatabase
except Exception:  # driver not installed yet -> soft-disable
    GraphDatabase = None

_driver = None


def _get_driver():
    global _driver
    if config.GRAPH_DISABLED or GraphDatabase is None:
        return None
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
    return _driver


_warned_once = False

def _run(cypher: str, **params):
    global _warned_once
    d = _get_driver()
    if d is None:
        return []
    try:
        with d.session() as s:
            return list(s.run(cypher, **params))
    except Exception as e:
        # Neo4j being unreachable should degrade the graph feature, not take
        # down the whole request. This also covers the case where
        # GRAPH_DISABLED didn't get picked up correctly for some environment-
        # specific reason -- the app stays usable either way.
        if not _warned_once:
            print(f"[graph] Neo4j unavailable, continuing without graph features: {e}")
            _warned_once = True
        return []


# ---- writes ---------------------------------------------------------------
def upsert_template(template_id: str, version: int):
    _run(
        """MERGE (t:Template {id:$id})
           SET t.name=$id, t.version=$version""",
        id=template_id,
        version=version,
    )


def upsert_prompt(prompt_id: str, version: int, template_id: str | None,
                  pinned_version: int | None):
    _run(
        """MERGE (p:Prompt {id:$id})
           SET p.name=$id, p.version=$version""",
        id=prompt_id,
        version=version,
    )
    if template_id:
        _run(
            """MATCH (p:Prompt {id:$pid}), (t:Template {id:$tid})
               MERGE (p)-[r:USES_TEMPLATE]->(t)
               SET r.pinned_version=$pinned""",
            pid=prompt_id,
            tid=template_id,
            pinned=pinned_version,
        )


def set_status(prompt_id: str, score: float, status: str):
    _run(
        "MATCH (p:Prompt {id:$id}) SET p.score=$score, p.status=$status",
        id=prompt_id,
        score=score,
        status=status,
    )


def set_pin(prompt_id: str, template_id: str, pinned_version: int):
    _run(
        """MATCH (p:Prompt {id:$pid})-[r:USES_TEMPLATE]->(t:Template {id:$tid})
           SET r.pinned_version=$pinned""",
        pid=prompt_id,
        tid=template_id,
        pinned=pinned_version,
    )


# ---- reads (for the UI) ---------------------------------------------------
def blast_radius(template_id: str) -> list[dict]:
    """Every prompt downstream of a template, with its current colour -- this is
    exactly what the front-end draws when a template is edited."""
    rows = _run(
        """MATCH (p:Prompt)-[r:USES_TEMPLATE]->(t:Template {id:$tid})
           RETURN p.id AS id, p.score AS score, p.status AS status,
                  r.pinned_version AS pinned""",
        tid=template_id,
    )
    if rows:
        return [dict(r) for r in rows]

    from . import db

    return [
        {
            "id": p["id"],
            "score": p["score"],
            "status": p["status"],
            "pinned": p["pinned_template_version"],
        }
        for p in db.prompts_using_template(template_id)
    ]


def snapshot() -> dict:
    """Whole graph, for rendering."""
    nodes = _run(
        """MATCH (n) RETURN labels(n)[0] AS type, n.id AS id,
                  n.status AS status, n.score AS score, n.version AS version"""
    )
    edges = _run(
        """MATCH (a)-[r]->(b)
           RETURN a.id AS source, b.id AS target, type(r) AS type,
                  r.pinned_version AS pinned"""
    )
    if nodes or edges:
        return {"nodes": [dict(n) for n in nodes], "edges": [dict(e) for e in edges]}

    from . import db

    templates = [
        {
            "type": "Template",
            "id": t["id"],
            "status": None,
            "score": None,
            "version": t["version"],
        }
        for t in db.latest_templates()
    ]
    prompts = [
        {
            "type": "Prompt",
            "id": p["id"],
            "status": p["status"],
            "score": p["score"],
            "version": p["version"],
        }
        for p in db.latest_prompts()
    ]
    fallback_edges = [
        {
            "source": p["id"],
            "target": p["template_id"],
            "type": "USES_TEMPLATE",
            "pinned": p["pinned_template_version"],
        }
        for p in db.latest_prompts()
        if p["template_id"]
    ]
    return {"nodes": templates + prompts, "edges": fallback_edges}
