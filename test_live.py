"""One-off diagnostic: times a single real run_prompt call and a single real
judge_presence call so we can see exactly how long Sarvam is taking, instead
of waiting blind on the full ~16-call /demo/seed chain.

Run from the PromptOps folder with your venv active:
    python test_live.py
"""
import sys
import time

sys.path.insert(0, ".")
from promptops import config, llm  # noqa: E402

print(f"MODE={config.LLM_MODE}  MODEL={config.SARVAM_MODEL}  "
      f"KEY_SET={bool(config.SARVAM_API_KEY)}  "
      f"KEY_PREFIX={config.SARVAM_API_KEY[:6]!r}")

if config.LLM_MODE != "sarvam":
    print("PROMPTOPS_LLM_MODE is not 'sarvam' -- nothing real will be tested.")
    sys.exit(1)
if not config.SARVAM_API_KEY:
    print("SARVAM_API_KEY is empty -- .env wasn't picked up (did you restart uvicorn / this shell?).")
    sys.exit(1)

print("\n--- calling run_prompt (real Sarvam chat call) ---")
t0 = time.time()
try:
    out = llm.run_prompt(
        "You are a helpful assistant. Always end your answer by telling the "
        "user this is not financial advice and not a recommendation to buy or sell.",
        "Should I buy TCS stock?",
    )
    print(f"OK in {time.time() - t0:.1f}s -> {out!r}")
except Exception as e:
    print(f"FAILED after {time.time() - t0:.1f}s -> {type(e).__name__}: {e}")
    sys.exit(1)

print("\n--- calling judge_presence (real Sarvam structured-output call) ---")
t0 = time.time()
try:
    result = llm.judge_presence(out, ["a disclosure that this isn't financial advice"])
    print(f"OK in {time.time() - t0:.1f}s -> {result}")
except Exception as e:
    print(f"FAILED after {time.time() - t0:.1f}s -> {type(e).__name__}: {e}")
    sys.exit(1)

print("\nBoth real calls completed. If /demo/seed still hangs longer than "
      "~10x this per-call time, something else is wrong -- otherwise it's "
      "just the 16 sequential calls adding up.")
