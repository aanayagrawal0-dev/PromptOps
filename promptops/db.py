"""Relational source of truth for *content* and *scores*.

Neo4j owns relationships and display state; this owns the actual text of every
prompt/template version and the score each version earned. SQLite here purely to
keep the demo zero-setup -- swap the connection for Postgres and the schema is
unchanged (all standard SQL).
"""
import json
import sqlite3
from contextlib import contextmanager

from . import config


@contextmanager
def _conn():
    c = sqlite3.connect(config.SQLITE_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS templates (
                id          TEXT NOT NULL,
                version     INTEGER NOT NULL,
                text        TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, version)
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id          TEXT NOT NULL,
                version     INTEGER NOT NULL,
                body        TEXT NOT NULL,          -- may contain {template_id} refs
                template_id TEXT,                   -- shared template it depends on
                dataset     TEXT NOT NULL,          -- JSON eval dataset
                score       REAL,                   -- last computed score
                status      TEXT,                   -- green/amber/red
                -- which template version THIS prompt version is pinned to.
                -- Option B lives here: change the pin -> resolved text changes.
                pinned_template_version INTEGER,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id, version)
            );

            -- Third dependency tier, above prompts: a real downstream
            -- consumer (an app/agent/pipeline) that calls one or more
            -- tracked prompts. Deliberately has no score/status column --
            -- its health is never stored, only ever computed live from
            -- whatever its prompts' current status actually is (see
            -- pipeline_status() in main.py), so it can never drift into
            -- showing a stale badge after a prompt underneath it changes.
            CREATE TABLE IF NOT EXISTS pipelines (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pipeline_prompts (
                pipeline_id TEXT NOT NULL,
                prompt_id   TEXT NOT NULL,
                PRIMARY KEY (pipeline_id, prompt_id)
            );

            -- One row per real regression the gate actually caught and acted
            -- on (auto-rollback or flagged for review) -- powers the
            -- Regressions page with real history instead of static mockup
            -- rows. duration_ms is real, measured wall-clock time for that
            -- prompt's re-eval + (if applicable) rollback, not a guess.
            CREATE TABLE IF NOT EXISTS regression_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id   TEXT NOT NULL,
                template_id TEXT NOT NULL,
                prev_score  REAL,
                new_score   REAL,
                status      TEXT NOT NULL,
                action      TEXT NOT NULL,          -- auto_rollback | flag_manual
                explanation TEXT,
                duration_ms INTEGER,
                -- the actual template version an auto_rollback repinned to
                -- (NULL for flag_manual, which doesn't roll anything back) --
                -- stored so the UI can say "rolled back to v1" honestly
                -- instead of guessing/hardcoding a version number.
                rolled_back_to_version INTEGER,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


# ---- templates ------------------------------------------------------------
def add_template_version(template_id: str, text: str) -> int:
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(version) AS v FROM templates WHERE id=?", (template_id,)
        ).fetchone()
        version = (row["v"] or 0) + 1
        c.execute(
            "INSERT INTO templates (id, version, text) VALUES (?,?,?)",
            (template_id, version, text),
        )
        return version


def get_template_text(template_id: str, version: int) -> str | None:
    with _conn() as c:
        row = c.execute(
            "SELECT text FROM templates WHERE id=? AND version=?",
            (template_id, version),
        ).fetchone()
        return row["text"] if row else None


def latest_template_version(template_id: str) -> int | None:
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(version) AS v FROM templates WHERE id=?", (template_id,)
        ).fetchone()
        return row["v"]


# ---- prompts --------------------------------------------------------------
def add_prompt_version(
    prompt_id: str,
    body: str,
    dataset: list,
    template_id: str | None = None,
    pinned_template_version: int | None = None,
) -> int:
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(version) AS v FROM prompts WHERE id=?", (prompt_id,)
        ).fetchone()
        version = (row["v"] or 0) + 1
        c.execute(
            """INSERT INTO prompts
               (id, version, body, template_id, dataset, pinned_template_version)
               VALUES (?,?,?,?,?,?)""",
            (
                prompt_id,
                version,
                body,
                template_id,
                json.dumps(dataset),
                pinned_template_version,
            ),
        )
        return version


def get_prompt_version(prompt_id: str, version: int) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM prompts WHERE id=? AND version=?", (prompt_id, version)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["dataset"] = json.loads(d["dataset"])
        return d


def latest_prompt_version(prompt_id: str) -> int | None:
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(version) AS v FROM prompts WHERE id=?", (prompt_id,)
        ).fetchone()
        return row["v"]


def set_prompt_result(prompt_id: str, version: int, score: float, status: str):
    with _conn() as c:
        c.execute(
            "UPDATE prompts SET score=?, status=? WHERE id=? AND version=?",
            (score, status, prompt_id, version),
        )


def set_pin(prompt_id: str, version: int, template_version: int):
    """Repin a prompt version to a different template version.
    This is what a template-induced auto-rollback actually *does*."""
    with _conn() as c:
        c.execute(
            "UPDATE prompts SET pinned_template_version=? WHERE id=? AND version=?",
            (template_version, prompt_id, version),
        )


def unlink_template(prompt_id: str, version: int):
    """Remove a prompt version's dependency on a shared template. It keeps
    its own body and history -- it just no longer resolves against, or gets
    affected by edits to, any template."""
    with _conn() as c:
        c.execute(
            "UPDATE prompts SET template_id=NULL, pinned_template_version=NULL WHERE id=? AND version=?",
            (prompt_id, version),
        )


def reset_all():
    """Wipe every tracked prompt, template, pipeline, and logged regression
    event -- what a real 'Reset Graph' action needs before reseeding, not
    just reloading the page."""
    with _conn() as c:
        c.execute("DELETE FROM prompts")
        c.execute("DELETE FROM templates")
        c.execute("DELETE FROM pipelines")
        c.execute("DELETE FROM pipeline_prompts")
        c.execute("DELETE FROM regression_events")


def delete_prompt(prompt_id: str):
    """Remove every version of a prompt entirely -- for cleaning up a single
    junk registration, as opposed to reset_all()'s full wipe-and-reseed."""
    with _conn() as c:
        c.execute("DELETE FROM prompts WHERE id=?", (prompt_id,))


def log_regression_event(
    prompt_id: str, template_id: str, prev_score: float | None,
    new_score: float, status: str, action: str, explanation: str,
    duration_ms: int, rolled_back_to_version: int | None = None,
):
    """Record a regression the gate actually caught and acted on. Only
    called for real auto_rollback/flag_manual outcomes -- this is history,
    not every re-eval."""
    with _conn() as c:
        c.execute(
            """INSERT INTO regression_events
               (prompt_id, template_id, prev_score, new_score, status,
                action, explanation, duration_ms, rolled_back_to_version)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (prompt_id, template_id, prev_score, new_score, status,
             action, explanation, duration_ms, rolled_back_to_version),
        )


def list_regression_events(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM regression_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def prompts_using_template(template_id: str) -> list[dict]:
    """The blast radius: every prompt (latest version) that depends on a template."""
    with _conn() as c:
        rows = c.execute(
            """SELECT p.* FROM prompts p
               JOIN (SELECT id, MAX(version) AS mv FROM prompts GROUP BY id) latest
                 ON p.id = latest.id AND p.version = latest.mv
               WHERE p.template_id = ?""",
            (template_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["dataset"] = json.loads(d["dataset"])
            out.append(d)
        return out


def all_prompts() -> list[dict]:
    """Latest version of every prompt with its current score/status/pin.

    SQLite is the source of truth the UI reads from -- the Neo4j snapshot is
    empty whenever the graph is disabled (the default demo config)."""
    with _conn() as c:
        rows = c.execute(
            """SELECT p.* FROM prompts p
               JOIN (SELECT id, MAX(version) AS mv FROM prompts GROUP BY id) latest
                 ON p.id = latest.id AND p.version = latest.mv
               ORDER BY p.id"""
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["dataset"] = json.loads(d["dataset"])
            out.append(d)
        return out


def get_latest_template(template_id: str) -> dict | None:
    """Current text of a template, so the UI can load the real thing to edit."""
    with _conn() as c:
        row = c.execute(
            """SELECT id, version, text FROM templates
               WHERE id=? ORDER BY version DESC LIMIT 1""",
            (template_id,),
        ).fetchone()
        return dict(row) if row else None


def last_passing_template_version(prompt_id: str, template_id: str) -> int | None:
    """Most recent template version at which this prompt last scored >= threshold.
    Where auto-rollback pins back to. Falls back to v1 if no history."""
    # For the demo we track this via the prompt's own scored history; simplest
    # robust fallback is: the pin the prompt held the last time it was green.
    with _conn() as c:
        row = c.execute(
            """SELECT pinned_template_version FROM prompts
               WHERE id=? AND status=? ORDER BY version DESC LIMIT 1""",
            (prompt_id, config.GREEN),
        ).fetchone()
        if row and row["pinned_template_version"] is not None:
            return row["pinned_template_version"]
        return 1  # safe default: the original template version


# ---- pipelines (3rd tier: real downstream consumers of prompts) -----------
def add_pipeline(pipeline_id: str, name: str, prompt_ids: list) -> None:
    """Register (or re-register) a pipeline and its dependency links.
    Re-running with the same id replaces its prompt links rather than
    duplicating them, so re-seeding is idempotent."""
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO pipelines (id, name) VALUES (?,?)",
            (pipeline_id, name),
        )
        c.execute("DELETE FROM pipeline_prompts WHERE pipeline_id=?", (pipeline_id,))
        for pid in prompt_ids:
            c.execute(
                "INSERT OR IGNORE INTO pipeline_prompts (pipeline_id, prompt_id) VALUES (?,?)",
                (pipeline_id, pid),
            )


def all_pipelines() -> list[dict]:
    """Every pipeline with the ids of the prompts it actually depends on.
    No score/status here -- callers compute that live from those prompt ids'
    current state (see main.py's pipeline_status), so a pipeline can never
    show a badge that's out of sync with the prompts underneath it."""
    with _conn() as c:
        pipelines = c.execute("SELECT id, name FROM pipelines ORDER BY id").fetchall()
        out = []
        for p in pipelines:
            links = c.execute(
                "SELECT prompt_id FROM pipeline_prompts WHERE pipeline_id=?",
                (p["id"],),
            ).fetchall()
            out.append({
                "id": p["id"],
                "name": p["name"],
                "prompt_ids": [r["prompt_id"] for r in links],
            })
        return out


def delete_pipeline(pipeline_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM pipelines WHERE id=?", (pipeline_id,))
        c.execute("DELETE FROM pipeline_prompts WHERE pipeline_id=?", (pipeline_id,))
