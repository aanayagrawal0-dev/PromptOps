"""Central configuration. Everything env-overridable so the demo box and the
judge's laptop behave the same."""
import os

# ---- Scoring thresholds (the heart of the gating logic) -------------------
# Hard floor. Below this a version is considered broken -> auto rollback.
THRESHOLD = float(os.getenv("PROMPTOPS_THRESHOLD", "0.8"))

# ---- LLM mode -------------------------------------------------------------
# "mock"   -> deterministic offline outputs. Use this for the live demo so a
#             network/API hiccup can never break the run on stage.
# "sarvam" -> real Sarvam chat-completions calls.
LLM_MODE = os.getenv("PROMPTOPS_LLM_MODE", "mock")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-m")

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
