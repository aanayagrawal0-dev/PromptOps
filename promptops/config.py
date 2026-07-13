"""Central configuration. Everything env-overridable so the demo box and the
judge's laptop behave the same."""
import os


def _load_dotenv():
    """Tiny .env loader so keys in the project-root .env are picked up without
    pulling in a dependency. Real environment variables always win over the
    file (we only setdefault), so `SARVAM_API_KEY=... uvicorn ...` still works.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

# ---- Scoring thresholds (the heart of the gating logic) -------------------
# Hard floor. Below this a version is considered broken -> auto rollback.
THRESHOLD = float(os.getenv("PROMPTOPS_THRESHOLD", "0.8"))

# Ceiling for the amber ("flagged") band. At or above this a version is
# considered healthy. Must match the score bands shown in the dashboard
# Legend (Healthy > 0.95, Flagged 0.8-0.95, Broken < 0.8) -- the badge a user
# sees is only trustworthy if it's derived from the same bands as the copy
# that explains it.
HEALTHY_THRESHOLD = float(os.getenv("PROMPTOPS_HEALTHY_THRESHOLD", "0.95"))

# ---- LLM mode -------------------------------------------------------------
# "mock"   -> deterministic offline outputs. Use this for the live demo so a
#             network/API hiccup can never break the run on stage.
# "sarvam" -> real Sarvam chat-completions calls.
LLM_MODE = os.getenv("PROMPTOPS_LLM_MODE", "mock")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1")
# "sarvam-m" is deprecated. The chat-completions endpoint currently accepts
# only "sarvam-30b" (64K context) or "sarvam-105b" (128K context) -- the
# smaller model is the sane default for a one-sentence explanation.
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-30b")

# Cap on the answer-generating call (not the judge call -- its output is a
# tiny JSON array, never at risk of truncation). Without this, the model
# defaults to Sarvam's 2048-token ceiling and can write long enough answers
# that a required instruction near the end of the system prompt (e.g. "always
# end your answer with X") never actually gets generated before the response
# is cut off -- that's a truncation artifact, not the model ignoring the
# instruction. Keeping this low both fixes that and keeps eval calls cheap.
SARVAM_MAX_TOKENS = int(os.getenv("SARVAM_MAX_TOKENS", "400"))

# Language the regression explanation is generated in (the payoff line).
EXPLAIN_LANGUAGE = os.getenv("PROMPTOPS_EXPLAIN_LANGUAGE", "Hindi")

# ---- Stores ---------------------------------------------------------------
SQLITE_PATH = os.getenv("PROMPTOPS_DB", "promptops.db")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# If true, graph writes become no-ops so you can develop the eval/rollback
# logic before Neo4j is even running.
GRAPH_DISABLED = os.getenv("PROMPTOPS_GRAPH_DISABLED", "false").lower() == "true"

# ---- Status vocabulary ----------------------------------------------------
GREEN = "green"   # >= previous score: an improvement or a tie
AMBER = "amber"   # below previous but still >= threshold: flag, human decides
RED = "red"       # below threshold: auto rollback fires
