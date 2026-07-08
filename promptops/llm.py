"""LLM calls, used for two things:
  1. running a resolved prompt to produce an output (the thing we score)
  2. generating the regression explanation in an Indian language (the payoff)

`mock` mode is deterministic and offline -- use it on stage. `sarvam` mode hits
the real Sarvam chat-completions endpoint.
"""
import requests

from . import config


# --------------------------------------------------------------------------
# Running a resolved prompt
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
    """
    out = f"Regarding '{user_input}': here is a helpful response."
    sp = system_prompt.lower()
    if "not financial advice" in sp:
        out += " Please note this is not financial advice and not a recommendation to buy or sell."
    if "concise" in sp:
        out = f"Regarding '{user_input}': short answer."
    return out


def _sarvam_chat(system_prompt: str, user_input: str) -> str:
    resp = requests.post(
        f"{config.SARVAM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.SARVAM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.SARVAM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# Explaining a regression in an Indian language (Sarvam's real differentiator)
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
        f"You are a release engineer. In {language}, explain in ONE short "
        f"sentence, for a non-technical stakeholder, why an AI prompt broke. "
        f"Be plain and specific. Reply only in {language}."
    )
    return _sarvam_chat(system, facts)


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
