"""
Cache Test — Per Model
======================
Sends the SAME prompt to each model TWICE (bypassing local DiskCache)
and inspects the raw usage fields to confirm API-level cache activation.

Cache minimum thresholds (per provider docs):
  Anthropic Sonnet  via OpenRouter → 2,048 tokens
  Gemini 3.1 Flash/Pro via OpenRouter → model-dependent
  OpenAI GPT-5/mini via OpenRouter → 1,024 tokens (auto, no cache_control needed)

Run from repo root:
    python scripts/test_cache.py
    python scripts/test_cache.py --roles screener1 arbiter writer   # test specific roles
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import yaml
from openai import OpenAI

# ── long system prompt (~2,500 tokens) ──────────────────────────────────────
# Padded to clear Anthropic's 2,048-token minimum and OpenAI's 1,024-token
# minimum. Uses real systematic-review content so it's representative.

LONG_SYSTEM_PROMPT = """
You are an expert systematic review methodologist. Your task is to screen
research articles for inclusion in a meta-analysis on repetitive transcranial
magnetic stimulation (rTMS) for mild cognitive impairment (MCI).

== INCLUSION CRITERIA ==
1. Study population: adults (18+) with mild cognitive impairment (MCI),
   early-stage Alzheimer's disease, or subjective cognitive decline (SCD).
2. Intervention: repetitive TMS (rTMS) at any frequency (1 Hz, 10 Hz, 20 Hz)
   or theta-burst stimulation (TBS) targeting any cortical site.
3. Comparator: sham stimulation, waitlist control, or active comparator.
4. Study design: randomized controlled trial (RCT) or controlled clinical trial (CCT).
5. Outcomes: at least one standardized cognitive measure (MMSE, MoCA, ADAS-Cog,
   ACE-R, CAMCOG, CDR, NPI, FAQ, ADL, or neuropsychological battery).
6. Language: English, Chinese, Japanese, Korean, or European languages.

== EXCLUSION CRITERIA ==
1. Animal studies or in-vitro studies.
2. Case reports, case series, or uncontrolled observational studies.
3. Healthy participants only (no clinical population).
4. Interventions other than TMS (e.g., tDCS, ECT, DBS, pharmacotherapy only).
5. No cognitive outcome reported.
6. Full text not available and abstract insufficient for assessment.
7. Duplicate publication of same trial (retain most complete report).

== DECISION SCALE (5-point) ==
Use exactly one of these values:
  "most_likely_include"  : Clearly meets ALL criteria; no meaningful doubt
  "likely_include"       : Probably meets; minor/resolvable uncertainty
  "undecided"            : Genuinely cannot determine from available text
  "likely_exclude"       : Probably fails one or more criteria
  "most_likely_exclude"  : Clearly violates at least one exclusion criterion

== SCORING RULES ==
- Population (P):  wrong diagnosis, wrong age, or healthy only → exclude
- Intervention (I): non-TMS modality → exclude; missing modality info → undecided
- Comparison (C): study design often NOT stated in abstract → use undecided, NOT exclude
- Outcome (O): cognitive measure often mentioned; if absent → undecided at T/A stage
- Overall decision driven by the WORST clearly-failing criterion.
- Uncertainty in C alone (undecided design) should NOT cause exclusion at this stage.

== CONFIDENCE RUBRIC ==
1.0 = Certain; all criteria clearly assessable from available text
0.8 = High; minor uncertainty on one secondary criterion
0.6 = Moderate; one criterion unclear or borderline
0.4 = Low; multiple criteria unclear; abstract poorly reported
0.2 = Very low; critical fields absent (no disease, no intervention type)
0.0 = Cannot assess (non-English abstract with no translatable content)

== OUTPUT FORMAT ==
Respond ONLY with valid JSON:
{
  "study_id": "...",
  "population": "<5-point scale>",
  "intervention": "<5-point scale>",
  "comparison_design": "<5-point scale>",
  "decision": "<5-point scale>",
  "confidence": 0.0-1.0,
  "reason": "brief reason noting which criterion drove the decision",
  "uncertainty_flags": ["aspects unclear, or empty array"]
}

== IMPORTANT REMINDERS ==
- Bias toward inclusion: false negatives are worse than false positives at screening.
- Use "undecided" when information is genuinely absent, not as a lazy default.
- Only "most_likely_exclude" when a criterion is CLEARLY violated.
- If response cannot be parsed, system defaults to "likely_include" to preserve recall.
- You are an INDEPENDENT screener: do not assume what another screener decided.
- Respond ONLY with the JSON object above. No preamble, no explanation, no markdown.

== BACKGROUND CONTEXT ==
Mild cognitive impairment (MCI) represents a transitional state between normal
aging and dementia, affecting approximately 15-20% of adults over 65. rTMS is
a non-invasive brain stimulation technique that uses rapidly changing magnetic
fields to induce electrical currents in cortical neurons. High-frequency rTMS
(≥5 Hz) typically increases cortical excitability, while low-frequency rTMS
(≤1 Hz) is inhibitory. Theta-burst stimulation (TBS) delivers bursts of 3 pulses
at 50 Hz repeated at 5 Hz, with intermittent TBS (iTBS) being facilitatory.

Common stimulation targets include the dorsolateral prefrontal cortex (DLPFC),
inferior parietal lobule (IPL), posterior parietal cortex (PPC), and precuneus.
Some protocols use multi-site stimulation. Session parameters typically range
from 10 to 30 sessions, with 5 sessions per week. Sham stimulation uses either
a tilted coil or an active sham coil that produces similar sounds/sensations
without the magnetic field penetrating the cortex.

Primary cognitive outcomes frequently measured:
- Global cognition: MMSE (Mini-Mental State Examination), MoCA (Montreal
  Cognitive Assessment), ADAS-Cog (Alzheimer's Disease Assessment Scale –
  Cognitive subscale)
- Memory: AVLT (Auditory Verbal Learning Test), CVLT, WMS (Wechsler Memory Scale)
- Executive function: TMT-B, WCST, Stroop, digit span backward
- Neuropsychiatric: NPI (Neuropsychiatric Inventory), CSDD, GDS
- Functional: FAQ (Functional Activities Questionnaire), ADCS-ADL, CDR-SB
""".strip()

SHORT_USER_PROMPT = (
    'Evaluate: Title: "Repetitive TMS for MCI: an RCT". '
    'Abstract: "60 MCI patients randomised to 10Hz rTMS vs sham for 20 sessions. '
    'MMSE improved 2.3 points (p=0.03)." '
    'Respond with JSON only.'
)


# ── model-specific message builder ──────────────────────────────────────────

def build_messages(model_id: str, system_prompt: str, user_prompt: str) -> list:
    """Build messages with cache_control for Anthropic models.
    Mirrors base_agent.py: pads to 8,200 chars so Sonnet's 2,048-token
    cache minimum is reached before wrapping with cache_control."""
    if model_id.startswith("anthropic/"):
        # Mirrors base_agent.py padding logic
        _tok_est = max(1, len(system_prompt) // 4)
        _pad_tokens_needed = max(0, 2100 - _tok_est)
        if _pad_tokens_needed > 0:
            _lines_needed = (_pad_tokens_needed // 15) + 1
            _pad = "\n" + "\n".join(["#" + "─" * 78] * _lines_needed)
            system_prompt = system_prompt + _pad
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": user_prompt},
        ]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


def extract_cache_tokens(usage) -> dict:
    """Extract cache metrics from response usage regardless of provider."""
    if usage is None:
        return {"cache_read": 0, "cache_write": 0, "cached_openai": 0}

    # Anthropic via OpenRouter
    cr = getattr(usage, "cache_read_input_tokens",     0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0

    # OpenAI via OpenRouter
    details = getattr(usage, "prompt_tokens_details", None)
    cached_oai = getattr(details, "cached_tokens", 0) if details else 0

    return {"cache_read": cr, "cache_write": cw, "cached_openai": cached_oai}


def call_once(client: OpenAI, model_id: str, model_cfg: dict,
              messages: list, label: str) -> dict:
    """Make one API call and return usage + cache info."""
    kwargs = dict(
        model=model_id,
        messages=messages,
        max_tokens=64,                        # minimal output for test
        temperature=0.0,
    )

    reasoning_effort = model_cfg.get("reasoning_effort")
    if reasoning_effort:
        kwargs["extra_body"] = {"reasoning": {"effort": "minimal"}}  # fastest for test

    seed = model_cfg.get("seed")
    if seed is not None:
        kwargs["seed"] = seed

    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        return {"error": str(e), "label": label}
    elapsed = time.time() - t0

    usage = resp.usage
    cache = extract_cache_tokens(usage)
    actual_model = getattr(resp, "model", model_id)

    return {
        "label":         label,
        "actual_model":  actual_model,
        "input_tokens":  getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
        "cache_read":    cache["cache_read"],
        "cache_write":   cache["cache_write"],
        "cached_openai": cache["cached_openai"],
        "elapsed_s":     round(elapsed, 2),
    }


# ── result printer ───────────────────────────────────────────────────────────

STATUS = {
    "hit":     "✅ CACHE HIT",
    "write":   "📝 CACHE WRITE",
    "none":    "⬜ NO CACHE",
    "unknown": "❓ UNCLEAR",
    "error":   "❌ ERROR",
}


def interpret(call1: dict, call2: dict) -> str:
    if "error" in call2:
        return "error"
    cr2 = call2["cache_read"] + call2["cached_openai"]
    cw1 = call1.get("cache_write", 0) + call1.get("cache_read", 0)  # any cache activity on call1
    if cr2 > 0:
        return "hit"
    if call1.get("cache_write", 0) > 0:
        return "write"   # write happened but no hit yet
    # OpenAI: some models auto-cache without write tokens
    if call1.get("input_tokens", 0) > 1024:
        return "unknown"
    return "none"


def print_result(role: str, model_id: str, call1: dict, call2: dict) -> None:
    status = interpret(call1, call2)
    print(f"\n  {'─'*60}")
    print(f"  {role:20s} [{model_id}]")
    print(f"  {'─'*60}")

    if "error" in call1:
        print(f"  Call 1: ERROR — {call1['error']}")
        return
    if "error" in call2:
        print(f"  Call 1: OK")
        print(f"  Call 2: ERROR — {call2['error']}")
        return

    def fmt(c: dict, n: int):
        cr  = c['cache_read']
        cw  = c['cache_write']
        oai = c['cached_openai']
        cache_info = ""
        if cw > 0:
            cache_info += f" | cache_write={cw}"
        if cr > 0:
            cache_info += f" | cache_read={cr} ✅"
        if oai > 0:
            cache_info += f" | cached(OAI)={oai} ✅"
        return (f"  Call {n}: actual={c['actual_model']:45s} "
                f"in={c['input_tokens']:5d} out={c['output_tokens']:4d} "
                f"t={c['elapsed_s']:.1f}s{cache_info}")

    print(fmt(call1, 1))
    print(fmt(call2, 2))
    print(f"\n  Result → {STATUS[status]}")

    # Advice
    if status == "none":
        inp = call1.get("input_tokens", 0)
        print(f"  Advice: system prompt = ~{inp} tokens. ", end="")
        if model_id.startswith("anthropic/") and inp < 2048:
            print(f"Anthropic min is 2,048 — pad system prompt by ~{2048 - inp} more tokens.")
        elif inp < 1024:
            print(f"Min threshold is 1,024 — pad system prompt by ~{1024 - inp} more tokens.")
        else:
            print("Threshold met but no cache — model may not support caching via OpenRouter.")
    elif status == "write":
        print("  Note: cache WRITTEN on call 1 but call 2 was too fast / different enough.")
        print("  In production (same system prompt across many studies) this WILL hit.")
    elif status == "unknown":
        print("  Note: OpenAI auto-caches — no write tokens reported. Watch for cached_tokens in prod.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="*",
                        help="Roles to test (default: all unique models)")
    args = parser.parse_args()

    with open("config/models.yaml", "r") as f:
        config = yaml.safe_load(f)

    client = OpenAI(
        base_url=config["base_url"],
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    all_models = config["models"]

    # deduplicate by model_id — only test each unique model once
    seen_model_ids: dict[str, str] = {}   # model_id → first role name
    for role, cfg in all_models.items():
        mid = cfg["model_id"]
        if mid not in seen_model_ids:
            seen_model_ids[mid] = role

    if args.roles:
        test_roles = {r: all_models[r] for r in args.roles if r in all_models}
    else:
        # one representative role per unique model_id
        test_roles = {role: all_models[role] for role in seen_model_ids.values()}

    print(f"\n{'='*64}")
    print(f"  Cache Test — {len(test_roles)} models via OpenRouter")
    print(f"  System prompt size: {len(LONG_SYSTEM_PROMPT):,} chars")
    print(f"  (Anthropic min 2,048 tok ≈ 6,000 chars | OpenAI min 1,024 tok ≈ 3,000 chars)")
    print(f"{'='*64}")

    for role, cfg in test_roles.items():
        model_id = cfg["model_id"]
        messages = build_messages(model_id, LONG_SYSTEM_PROMPT, SHORT_USER_PROMPT)

        print(f"\n  Testing {role} [{model_id}]...", end=" ", flush=True)

        call1 = call_once(client, model_id, cfg, messages, "call1")
        time.sleep(1)   # brief pause between calls
        call2 = call_once(client, model_id, cfg, messages, "call2")

        print("done.")
        print_result(role, model_id, call1, call2)

    print(f"\n{'='*64}")
    print("  Test complete.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
