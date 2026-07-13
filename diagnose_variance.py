"""One-off: shows the REAL output and judge verdict for each of fund_summary's
two dataset rows, so we can see whether the model is skipping the disclosure
on one question, or the judge is being inconsistent about recognizing it that
IS there. Run from the PromptOps folder with venv active:

    python diagnose_variance.py
"""
import sys

sys.path.insert(0, ".")
from promptops import config, llm, resolver, seed  # noqa: E402

print(f"MODE={config.LLM_MODE}  MODEL={config.SARVAM_MODEL}\n")

system_prompt = "You are a helpful assistant. " + seed.DISCLAIMER_V1
print(f"SYSTEM PROMPT:\n{system_prompt}\n")

for row in seed._DATASET:
    print(f"--- INPUT: {row['input']!r} ---")
    output = llm.run_prompt(system_prompt, row["input"])
    print(f"REAL OUTPUT:\n{output}\n")
    concepts = row["checks"]["must_contain"]
    presence = llm.judge_presence(output, concepts)
    for i, concept in enumerate(concepts):
        verdict = presence.get(i)
        print(f"JUDGE VERDICT for {concept!r}: {verdict}")
    print()
