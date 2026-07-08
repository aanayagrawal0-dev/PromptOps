"""Option B: live template resolution.

A prompt body stores a *reference* like `{compliance_disclaimer}`, never the
template's text. At run time we look up whichever template version the prompt is
currently pinned to and splice its text in. Change the pin (or the template) and
the resolved prompt changes -- that is the whole reason a template edit can
silently alter a prompt nobody opened.
"""
import re

from . import db

_REF = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def resolve(prompt_row: dict, template_version: int | None = None) -> str:
    """Return the fully-resolved system prompt.

    template_version lets a caller resolve against a *hypothetical* version
    (e.g. the just-edited one) without repinning first -- used to test blast
    radius before committing a rollback decision.
    """
    body = prompt_row["body"]
    template_id = prompt_row.get("template_id")
    if not template_id:
        return body

    version = template_version
    if version is None:
        version = prompt_row.get("pinned_template_version") or db.latest_template_version(
            template_id
        )

    template_text = db.get_template_text(template_id, version) or ""

    def _sub(m):
        return template_text if m.group(1) == template_id else m.group(0)

    return _REF.sub(_sub, body)
