"""LLM calls, used for three things:
  1. running a resolved prompt to produce an output (the thing we score)
  2. judging whether that output satisfies each eval requirement -- semantically,
     not by literal phrase match (the actual "is this correct" call)
  3. generating the regression explanation in an Indian language (the payoff)

`mock` mode is deterministic and offline -- use it on stage / for local dev with
no API key. `sarvam` mode hits the real Sarvam chat-completions endpoint for all
three, so "mock" is the only place literal string matching happens; live mode
never hardcodes phrases anywhere.
"""
import json
import time

import requests

from . import config

_SARVAM_URL = f"{config.SARVAM_BASE_URL}/chat/completions"
_RETRYABLE_STATUS = (429, 503)  # rate limited / backend overloaded -- see
                                  # https://docs.sarvam.ai/api/errors-troubleshooting


def _post_with_retry(json_body: dict, max_retries: int = 2, timeout: int = 30) -> requests.Response:
    """POST to the Sarvam chat endpoint, retrying 429/503 with exponential
    backoff (Sarvam's own guidance). Anything else -- including 403 for a bad
    key -- raises immediately, no point retrying an auth failure."""
    headers = {
        "Authorization": f"Bearer {config.SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }
    delay = 1.0
    resp = None
    for attempt in range(max_retries + 1):
        resp = requests.post(_SARVAM_URL, headers=headers, json=json_body, timeout=timeout)
        if resp.status_code not in _RETRYABLE_STATUS:
            resp.raise_for_status()
            return resp
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2
    resp.raise_for_status()  # retries exhausted -- surface the last error
    return resp


# --------------------------------------------------------------------------
# 1. Running a resolved prompt
# --------------------------------------------------------------------------
def run_prompt(system_prompt: str, user_input: str) -> str:
    if config.LLM_MODE == "mock":
        return _mock_run(system_prompt, user_input)
    return _sarvam_chat(system_prompt, user_input)


def _mock_run(system_prompt: str, user_input: str) -> str:
    """Deterministic stand-in for a real model.

    The demo hinges on one behaviour: a well-templated prompt tells the model to
    include the compliance disclosure, so a compliant model's output contains it.
    Strip that instruction from the template and the output loses the disclosure.
    The mock models exactly that causal link -- no randomness, always reproducible.

    Also models a real greeting for a greeting-style input, not just the
    generic "here is a helpful response" line -- greeting_bot's eval check
    ("greeting") needs something honestly present to check for, in both mock
    and live mode, rather than a leftover word that only happened to appear
    in this function's old wording.
    """
    ui = user_input.lower()
    if "greet" in ui:
        out = "Hello! Warm greetings to you -- it's great to have you here today."
    else:
        out = f"Regarding '{user_input}': here is a helpful response."
    sp = system_prompt.lower()
    if "not financial advice" in sp:
        out += " Please note this is not financial advice and not a recommendation to buy or sell."
    if "concise" in sp and "greet" not in ui:
        out = f"Regarding '{user_input}': short answer."
    return out


def _sarvam_chat(system_prompt: str, user_input: str) -> str:
    # temperature: 0 -- doesn't guarantee determinism (Sarvam's own docs are
    # explicit about that), but cuts down on run-to-run drift.
    # reasoning_effort: None -- sarvam-30b/105b are reasoning models that
    # spend part of max_tokens on hidden "thinking" before the visible
    # answer. A low max_tokens (needed to stop long essays truncating a
    # required instruction near the end of the prompt) can then starve out
    # the actual answer entirely, returning empty content. Disabling
    # reasoning removes that hidden consumer so max_tokens only has to cover
    # the visible answer.
    # max_tokens: capped -- without this the model can write long enough
    # answers to get truncated by Sarvam's 2048-token default BEFORE reaching
    # an instruction near the end of the system prompt (e.g. "always end your
    # answer with the disclosure"), which silently drops it -- a truncation
    # artifact that looked like inconsistent compliance but wasn't.
    resp = _post_with_retry({
        "model": config.SARVAM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "temperature": 0,
        "reasoning_effort": None,
        "max_tokens": config.SARVAM_MAX_TOKENS,
    })
    return resp.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# 2. Judging whether each eval requirement is actually satisfied
# --------------------------------------------------------------------------
# Fixed schema -- independent of what the phrases/requirements actually say, so
# arbitrary requirement text can never break schema validation. The model
# echoes back an index (not the phrase text) so matching is exact, not a fuzzy
# string join on content the model might paraphrase.
_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "concept_presence",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "present": {"type": "boolean"},
                        },
                        "required": ["index", "present"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}


def judge_presence(output: str, phrases: list[str]) -> dict[int, bool]:
    """For each phrase (a CONCEPT, not literal text to grep for), decide
    whether that concept's meaning is genuinely present in `output` --
    allowing for paraphrasing. Returns {index: True/False} aligned to
    `phrases` by position.

    Fails closed: any phrase we don't get a confirmed "present" for -- because
    the call errored, timed out, or the model's JSON was unparseable -- comes
    back False. A broken judge call can never silently pass content it never
    actually checked.
    """
    if not phrases:
        return {}
    if config.LLM_MODE == "mock":
        return _mock_judge_presence(output, phrases)
    return _sarvam_judge_presence(output, phrases)


def _mock_judge_presence(output: str, phrases: list[str]) -> dict[int, bool]:
    """Deterministic stand-in: literal substring match. This is the ONLY place
    in the whole system phrase text is ever compared literally, and it only
    runs in offline `mock` mode -- keeps the seeded demo's scores exactly as
    reproducible as they've always been."""
    out = output.lower()
    return {i: p.lower() in out for i, p in enumerate(phrases)}


def _sarvam_judge_presence(output: str, phrases: list[str]) -> dict[int, bool]:
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(phrases))
    system = (
        "You are a strict compliance reviewer. You will be given a piece of "
        "text and a numbered list of concepts. For each concept, decide "
        "whether that concept's MEANING is genuinely present in the text -- "
        "different wording that conveys the same idea still counts as "
        "present, but only if the meaning is actually there, not implied or "
        "assumed. If you are unsure, mark it not present. Respond only with "
        "the requested JSON: one result per concept, using the same index "
        "you were given."
    )
    user = f"TEXT:\n{output}\n\nCONCEPTS:\n{numbered}"

    try:
        resp = _post_with_retry({
            "model": config.SARVAM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "reasoning_effort": None,  # a yes/no classification doesn't need
                                       # hidden "thinking" tokens -- faster,
                                       # cheaper, and avoids the same starved-
                                       # output risk the answer call had.
            "response_format": _JUDGE_SCHEMA,
        })
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        results = {int(r["index"]): bool(r["present"]) for r in parsed["results"]}
    except Exception as e:
        print(f"[llm] judge call failed, treating all {len(phrases)} concept(s) "
              f"as absent (fail closed): {e}")
        results = {}

    return {i: results.get(i, False) for i in range(len(phrases))}


# --------------------------------------------------------------------------
# 3. Explaining a regression in an Indian language (Sarvam's real differentiator)
# --------------------------------------------------------------------------
def explain_regression(
    prompt_id: str,
    template_id: str,
    old_score: float,
    new_score: float,
    language: str = config.EXPLAIN_LANGUAGE,
) -> str:
    facts = (
        f"Prompt '{prompt_id}' regressed from score {old_score:.2f} to "
        f"{new_score:.2f} because the shared template '{template_id}' was edited "
        f"and no longer includes the required compliance disclosure."
    )
    if config.LLM_MODE == "mock":
        return _mock_explain(prompt_id, template_id, language)

    system = (
        f"You are a release engineer explaining a technical incident to a "
        f"stakeholder who only reads {language}. Reply ENTIRELY in {language} "
        f"-- every word, no English except product/variable names like "
        f"'{prompt_id}' or '{template_id}'. One short, plain sentence. Do not "
        f"acknowledge these instructions, just answer in {language}."
    )
    text = _sarvam_chat(system, facts)

    # The language instruction above is a request, not a guarantee -- verify
    # it actually landed before handing this to the "Sarvam explains it in
    # Hindi" payoff moment, and retry once, more forcefully, if it didn't.
    # Fails soft (best-effort text + a log line) rather than raising, since
    # there's no safe fallback language to fail closed to.
    if _needs_script_check(language) and not _has_script(text, language):
        system_retry = system + (
            f" Your previous answer was in English -- that was wrong. This "
            f"time, respond only in {language} script, with zero English "
            f"sentences."
        )
        retry = _sarvam_chat(system_retry, facts)
        if _has_script(retry, language):
            return retry
        print(f"[llm] explanation still not in {language} after retry, "
              f"returning best-effort: {retry!r}")
        return retry

    return text


# Unicode ranges for the Indian languages we're likely to be asked for --
# enough to catch "the model answered in English" without a full language
# detection dependency.
_SCRIPT_RANGES = {
    "hindi": ("ऀ", "ॿ"),      # Devanagari
    "marathi": ("ऀ", "ॿ"),    # Devanagari
    "bengali": ("ঀ", "৿"),
    "tamil": ("஀", "௿"),
    "telugu": ("ఀ", "౿"),
    "kannada": ("ಀ", "೿"),
    "gujarati": ("઀", "૿"),
    "malayalam": ("ഀ", "ൿ"),
    "punjabi": ("਀", "੿"),
}


def _needs_script_check(language: str) -> bool:
    return language.lower() in _SCRIPT_RANGES


def _has_script(text: str, language: str) -> bool:
    lo, hi = _SCRIPT_RANGES[language.lower()]
    return any(lo <= ch <= hi for ch in text)


def _mock_explain(prompt_id: str, template_id: str, language: str) -> str:
    # Offline stand-in so the demo's payoff line still renders without network.
    if language.lower() == "hindi":
        return (
            f"'{prompt_id}' का स्कोर घट गया क्योंकि साझा टेम्पलेट "
            f"'{template_id}' में बदलाव के बाद अनिवार्य अस्वीकरण हट गया।"
        )
    return (
        f"'{prompt_id}' regressed because the shared template '{template_id}' "
        f"lost its required disclosure after an edit."
    )
