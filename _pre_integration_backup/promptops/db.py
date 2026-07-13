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
