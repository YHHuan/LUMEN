# PORT_NOTES — V4 → v3 (`stabilize` branch)

Status of porting the three real improvements from V4 (`lumen-code`, private) into
the published v3 (`stabilize`). Guiding rule: port only what can be verified
**without** running the pipeline or spending money; leave anything that needs a
live API key / real run for the user, clearly marked.

Legend: ✅ ported + unit-tested · 🟡 skeleton ported (off by default) · ⏭️ deferred to user

---

## 1. Quality bucketing (ROBINS-I / RoB2 → tiers) — ✅ / ⏭️

- ✅ **Ported (deterministic, tested):** `lumen/tools/quality/quality_bucket.py`
  - `robins_i_overall()` — closed-form ROBINS-I overall (worst-domain; ≥3 Serious → Critical), the non-randomised-study path v3 lacked. Mirrors V4 `quality/robins_i_algorithm.py:298`.
  - `quality_bucket()` / `bucket_studies()` — map any RoB2 (`low|some_concerns|high`) **or** ROBINS-I (`Low|Moderate|Serious|Critical`) overall into `LOW_ROB|MODERATE_ROB|HIGH_ROB`. Case-insensitive, so it can never suffer a vocabulary mismatch.
  - `grade_rob_downgrade()` — V4's rule (`>0.5`→2, `≥0.25` or any-high→1) from V4 `nodes/quality_node.py:152`.
  - Tests: `tests/unit/test_quality_bucket.py`.
- ⏭️ **Deferred to user (would change published GRADE output → needs a real run to validate):**
  - Wiring `grade_rob_downgrade` into the live GRADE path (`lumen/agents/quality_node.py:_assess_grade_per_outcome`, currently `prop_high ≥ 0.5 / ≥ 0.25`). V4's "any high → downgrade 1" is stricter; adopting it shifts certainty ratings, so it is offered as a tool, not auto-applied.
  - The **LLM** ROBINS-I domain assessor + design→tool router (V4 `quality/router.py`, `robins_i_assessor.py`) — needs an API key. v3 only assesses RoB2 (RCTs) via LLM today.

## 2. Prompt caching — 🟡 / ⏭️

- 🟡 **Skeleton ported, OFF by default:** `lumen/infra/llm_cache.py` (`PromptCache`, `make_cache_key`)
  - Content-addressed SHA-256 disk cache (mirrors V4 `infra/llm_cache.py`, rewritten fresh — no code/secrets copied).
  - Wired into `ModelRouter.call()` **gated** behind `is_caching_enabled(config)` (`lumen/core/config.py`). Default off → the router behaves exactly as v3 did (verified: `test_llm_cache.py::test_caching_off_calls_llm_every_time`).
  - Honest-cost fields prepared: on a cache hit the router reports `cached_input_tokens` and `cost=0`; `CostTracker.summary()` exposes `grand_total_cached_tokens` / `caching_enabled` / `cost_basis`.
  - **Evidence-run always disables it** (`--evidence-run`), preventing V4's real footgun (v4.0 defaulted cache ON → silent reuse). Tested: `test_evidence_run_disables_cache`.
  - Tests: `tests/unit/test_llm_cache.py`.
- ⏭️ **Deferred to user (needs a live API run to validate):**
  - Confirming real cache hit/miss + $ savings against a provider.
  - Anthropic **ephemeral** prompt caching (`cache_control: ephemeral` on the system block for prompts ≥ ~1024 tokens; V4 `infra/llm_router.py:106`). This is a LiteLLM/Anthropic-specific optimisation that only shows value on real calls; not added to keep the change verifiable offline. Enable per V4 once validated with a key.

## 3. Direct-provider routing — ✅ / ⏭️

- ✅ **Already present in v3 + now confirmed/tested.** v3's `ModelRouter` passes each tier's model string **straight to LiteLLM**, which routes by prefix — so any tier can name any provider directly (`anthropic/…`, `gemini/…`, `openai/…`) or via `openrouter/…`. No tier is bound to a provider in code (the openrouter prefixes live only in `configs/models.yaml`, which is data).
  - Tests: `tests/unit/test_provider_routing.py` (each tier → its named provider; custom tier names; openrouter prefix). Also `tests/unit/test_router.py`'s fixture already used direct `gemini/…`, `anthropic/…`.
  - `configs/models.yaml` now documents a direct-provider example.
- ⏭️ **Deferred to user (design choice / needs keys):**
  - V4's *explicit* per-provider clients + base_urls + provider-selection table (`provider_for_tier`: FAST→google, SMART/STRATEGIC→anthropic, openrouter fallback; V4 `infra/llm_router.py:169`, `core/config.py:108`). In v3 this is unnecessary — LiteLLM reads `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` from the env automatically. If the user wants V4's exact routing table, switch `models.yaml` primaries to direct prefixes and set the per-provider keys.

---

## The V4 "weight vocabulary" bug — checked, NOT present in v3

V4 has a real dead-guard: `strategy_generator.py:290` defaults screening-question
`weight` to the literal `"important"`, but `screening/decision.py:25`
`WEIGHT_VALUES` only accepts `{CRITICAL, HIGH, MEDIUM}`, so the hard-exclusion
guards (`decision.py:152/255/331`) never fire.

**v3 has no weighted-question screening guard at all** — that whole
`decision.py` / `WEIGHT_VALUES` / `questions.py` machinery is V4-only. v3 screening
is pure LLM decision + confidence routing (`screener` → `arbiter` → `resolve_screening`),
so the bug cannot exist here. The one severity-routed guard v3 *does* have
(`route_after_quality`, critical anomaly → re-extract) uses a **matched**
vocabulary (`statistician` emits `{critical, warning, info}`; the router escalates
on `critical`). Locked by `tests/unit/test_anomaly_guard.py`.

Separately, a v3 latent bug of the **same class** (producer/consumer key mismatch)
was found and fixed: `cost_report.py` read `stats["tokens"]` while the real
`CostTracker` emits `input_tokens`/`output_tokens` → the report showed 0 tokens on
real data (masked by a wrong-shaped test fixture). Fixed robustly + tested
(`tests/unit/test_honest_cost.py::TestCostReportTokenKeyFix`).
