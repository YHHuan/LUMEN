"""
Statistician agent: data profile → analysis plan → execute → anomaly detection → interpretation.

Combines v2's analysis_planner + statistician into a single agent.
Uses Sprint 2's deterministic functions for all calculations.
"""
from __future__ import annotations

import json
import structlog

from lumen.agents.base import BaseAgent, LumenParseError
from lumen.tools.statistics.effect_sizes import (
    hedges_g, mean_difference, log_risk_ratio, odds_ratio, risk_difference,
)
from lumen.tools.statistics.meta_analysis import (
    random_effects_meta, subgroup_meta, leave_one_out,
)
from lumen.tools.statistics.publication_bias import egger_test, trim_and_fill
from lumen.tools.statistics.heterogeneity import cochran_q, i_squared

logger = structlog.get_logger()

# Anomaly thresholds
I2_HIGH_THRESHOLD = 90.0
WEIGHT_DOMINANCE_THRESHOLD = 0.40
MIN_K_WARNING = 3
EGGER_MIN_K = 10


class StatisticianAgent(BaseAgent):
    tier = "smart"
    agent_name = "statistician"
    prompt_file = "statistician_plan.yaml"

    def analyze(self, harmonized_data: list[dict], pico: dict,
                quality_assessments: dict | None = None) -> dict:
        """
        Full 5-step analysis pipeline.

        Step 1: Data profile
        Step 2: Analysis plan (LLM-assisted)
        Step 3: Execute (deterministic — Sprint 2 functions only)
        Step 4: Anomaly detection
        Step 5: Interpretation (LLM-assisted)
        """
        # Step 1: Data profile
        profile = self._data_profile(harmonized_data)
        logger.info("statistician_step1_profile", n_outcomes=len(profile["outcomes"]))

        if not profile["outcomes"]:
            return {
                "analysis_plan": {},
                "statistics_results": {},
                "anomaly_flags": [],
                "interpretations": {},
            }

        # Step 2: Analysis plan (LLM)
        plan = self._make_plan(profile, pico)
        logger.info("statistician_step2_plan", n_planned=len(plan.get("outcomes", [])))

        # Step 3: Execute analyses
        results = self._execute(plan, harmonized_data)
        logger.info("statistician_step3_execute", n_results=len(results))

        # Step 4: Anomaly detection
        anomalies = self._detect_anomalies(results, plan, pico)
        logger.info("statistician_step4_anomalies", n_flags=len(anomalies))

        # Step 5: Interpretation (LLM)
        interpretations = self._interpret(results, anomalies, pico,
                                          quality_assessments)
        logger.info("statistician_step5_interpret")

        return {
            "data_profile": profile,
            "analysis_plan": plan,
            "statistics_results": results,
            "anomaly_flags": anomalies,
            "interpretations": interpretations,
        }

    @staticmethod
    def _data_profile(harmonized_data: list[dict]) -> dict:
        """Step 1: Profile the harmonized data to determine what analyses to run."""
        outcomes: dict[str, dict] = {}

        for study in harmonized_data:
            study_id = study.get("study_id", "unknown")
            items = study.get("extractions", [study] if "outcome_name" in study else [])

            for item in items:
                name = item.get("canonical_outcome", item.get("outcome_name", ""))
                if not name:
                    continue

                if name not in outcomes:
                    outcomes[name] = {
                        "name": name,
                        "studies": [],
                        "measure_types": set(),
                        "has_binary": False,
                        "has_continuous": False,
                        "subgroup_vars": set(),
                    }

                entry = outcomes[name]
                entry["studies"].append(study_id)

                # Detect binary vs continuous
                arm1 = item.get("arm1", {})
                if arm1.get("events") is not None and arm1.get("total") is not None:
                    entry["has_binary"] = True
                    entry["measure_types"].add("binary")
                if arm1.get("mean") is not None and arm1.get("sd") is not None:
                    entry["has_continuous"] = True
                    entry["measure_types"].add("continuous")

                # Detect subgroup info
                for key in item:
                    if key.startswith("subgroup_"):
                        entry["subgroup_vars"].add(key)

        # Convert sets to lists for JSON serialization
        profile_outcomes = []
        for o in outcomes.values():
            profile_outcomes.append({
                "name": o["name"],
                "k": len(o["studies"]),
                "study_ids": o["studies"],
                "measure_types": list(o["measure_types"]),
                "has_binary": o["has_binary"],
                "has_continuous": o["has_continuous"],
                "subgroup_vars": list(o["subgroup_vars"]),
            })

        return {
            "n_outcomes": len(profile_outcomes),
            "outcomes": profile_outcomes,
            "total_studies": len({s for o in outcomes.values() for s in o["studies"]}),
        }

    def _make_plan(self, profile: dict, pico: dict) -> dict:
        """Step 2: LLM generates analysis plan based on data profile."""
        user_content = (
            f"## PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## Data Profile\n{json.dumps(profile, indent=2)}\n\n"
            "Based on the data profile and PICO, create an analysis plan."
        )
        messages = self._build_messages(user_content)

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase="statistician_plan",
            )
            return self._parse_json(response, retry_messages=messages,
                                    phase="statistician_plan")
        except LumenParseError:
            logger.warning("statistician_plan_fallback",
                           msg="LLM plan failed, using defaults")
            return self._default_plan(profile)

    @staticmethod
    def _default_plan(profile: dict) -> dict:
        """Fallback plan when LLM planning fails."""
        outcomes = []
        for o in profile["outcomes"]:
            k = o["k"]
            measure = "continuous" if o["has_continuous"] else "binary"
            outcomes.append({
                "name": o["name"],
                "measure": measure,
                "effect_size": "SMD" if measure == "continuous" else "logOR",
                "model": "random_effects",
                "method": "REML" if k >= 3 else "DL",
                "apply_hksj": True,
                "k": k,
                "subgroup_variables": o.get("subgroup_vars", []),
                "sensitivity_analyses": ["leave_one_out"] if k >= 3 else [],
                "run_egger": k >= EGGER_MIN_K,
                "run_trim_and_fill": k >= EGGER_MIN_K,
                "notes": "",
            })
        return {"outcomes": outcomes, "global_notes": "Default plan (LLM fallback)"}

    def _execute(self, plan: dict, harmonized_data: list[dict]) -> dict:
        """Step 3: Execute analyses deterministically using Sprint 2 functions."""
        results = {}

        for outcome_plan in plan.get("outcomes", []):
            name = outcome_plan["name"]
            effect_size_type = outcome_plan.get("effect_size", "SMD")
            method = outcome_plan.get("method", "REML")
            apply_hksj = outcome_plan.get("apply_hksj", True)

            # Collect data for this outcome
            effects, ses, study_labels = self._collect_outcome_data(
                harmonized_data, name, effect_size_type,
            )

            if len(effects) == 0:
                results[name] = {"error": "No data available", "k": 0}
                continue

            outcome_result: dict = {"k": len(effects), "effect_size_type": effect_size_type}

            # Primary meta-analysis
            try:
                meta = random_effects_meta(
                    effects, ses, method=method, apply_hksj=apply_hksj,
                )
                outcome_result["meta"] = meta
            except (ValueError, RuntimeError) as e:
                outcome_result["meta"] = {"error": str(e)}
                results[name] = outcome_result
                continue

            # Sensitivity: leave-one-out
            if "leave_one_out" in outcome_plan.get("sensitivity_analyses", []):
                if len(effects) >= 2:
                    try:
                        loo = leave_one_out(effects, ses, labels=study_labels,
                                            method=method, apply_hksj=apply_hksj)
                        outcome_result["leave_one_out"] = loo
                    except ValueError:
                        pass

            # Subgroup analysis
            for subgroup_var in outcome_plan.get("subgroup_variables", []):
                groups = self._collect_subgroup_labels(
                    harmonized_data, name, subgroup_var,
                )
                if groups and len(set(groups)) > 1:
                    try:
                        sub = subgroup_meta(effects, ses, groups,
                                            method=method, apply_hksj=apply_hksj)
                        outcome_result.setdefault("subgroup_analyses", {})[subgroup_var] = sub
                    except ValueError:
                        pass

            # Publication bias
            if outcome_plan.get("run_egger") and len(effects) >= 3:
                try:
                    outcome_result["egger"] = egger_test(effects, ses)
                except ValueError:
                    pass

            if outcome_plan.get("run_trim_and_fill") and len(effects) >= 3:
                try:
                    outcome_result["trim_and_fill"] = trim_and_fill(effects, ses)
                except ValueError:
                    pass

            results[name] = outcome_result

        return results

    @staticmethod
    def _impute_arm(arm: dict) -> dict:
        """Try to fill missing SD from SE+n, or from CI+n.

        Modifies a *copy* — original dict is not mutated.
        """
        arm = dict(arm)
        n = arm.get("n")
        # SD from SE: SD = SE * sqrt(n)
        if arm.get("sd") is None and arm.get("se") is not None and n:
            import math
            arm["sd"] = arm["se"] * math.sqrt(n)
        # SD from 95% CI: SD ≈ sqrt(n) * (upper - lower) / 3.92
        if arm.get("sd") is None and arm.get("ci_lower") is not None and arm.get("ci_upper") is not None and n:
            import math
            arm["sd"] = math.sqrt(n) * (arm["ci_upper"] - arm["ci_lower"]) / 3.92
        return arm

    @staticmethod
    def _collect_outcome_data(harmonized_data: list[dict], outcome_name: str,
                              effect_size_type: str) -> tuple[list, list, list]:
        """Extract effect sizes and SEs for a specific outcome."""
        effects = []
        ses = []
        labels = []

        for study in harmonized_data:
            study_id = study.get("study_id", "unknown")
            items = study.get("extractions", [study] if "outcome_name" in study else [])

            for item in items:
                canonical = item.get("canonical_outcome", item.get("outcome_name", ""))
                if canonical != outcome_name:
                    continue

                arm1 = StatisticianAgent._impute_arm(item.get("arm1", {}))
                arm2 = StatisticianAgent._impute_arm(item.get("arm2", {}))

                # Skip arms where all numeric fields are None (no-PDF placeholder)
                if all(arm1.get(k) is None for k in ("mean", "sd", "events")):
                    logger.info("effect_size_skipped_no_data",
                                study_id=study_id, outcome=outcome_name)
                    continue

                try:
                    if effect_size_type == "SMD":
                        r = hedges_g(
                            n1=arm1["n"], mean1=arm1["mean"], sd1=arm1["sd"],
                            n2=arm2["n"], mean2=arm2["mean"], sd2=arm2["sd"],
                        )
                        effects.append(r["g"])
                        ses.append(r["se"])
                    elif effect_size_type == "MD":
                        r = mean_difference(
                            mean1=arm1["mean"], sd1=arm1["sd"], n1=arm1["n"],
                            mean2=arm2["mean"], sd2=arm2["sd"], n2=arm2["n"],
                        )
                        effects.append(r["md"])
                        ses.append(r["se"])
                    elif effect_size_type == "logRR":
                        r = log_risk_ratio(
                            a=arm1["events"], n1=arm1["total"],
                            c=arm2["events"], n2=arm2["total"],
                        )
                        effects.append(r["log_rr"])
                        ses.append(r["se"])
                    elif effect_size_type == "logOR":
                        r = odds_ratio(
                            a=arm1["events"],
                            b=arm1["total"] - arm1["events"],
                            c=arm2["events"],
                            d=arm2["total"] - arm2["events"],
                        )
                        effects.append(r["log_or"])
                        ses.append(r["se"])
                    elif effect_size_type == "RD":
                        r = risk_difference(
                            a=arm1["events"], n1=arm1["total"],
                            c=arm2["events"], n2=arm2["total"],
                        )
                        effects.append(r["rd"])
                        ses.append(r["se"])
                    else:
                        continue

                    labels.append(study_id)
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                    logger.warning("effect_size_calc_failed",
                                   study_id=study_id, outcome=outcome_name,
                                   reason=str(exc),
                                   arm1_keys=list(arm1.keys()),
                                   arm2_keys=list(arm2.keys()))
                    continue

        return effects, ses, labels

    @staticmethod
    def _collect_subgroup_labels(harmonized_data: list[dict],
                                 outcome_name: str,
                                 subgroup_var: str) -> list[str]:
        """Collect subgroup labels for a specific outcome and variable."""
        labels = []
        for study in harmonized_data:
            items = study.get("extractions", [study] if "outcome_name" in study else [])
            for item in items:
                canonical = item.get("canonical_outcome", item.get("outcome_name", ""))
                if canonical != outcome_name:
                    continue
                labels.append(str(item.get(subgroup_var, "unknown")))
        return labels

    @staticmethod
    def _detect_anomalies(results: dict, plan: dict, pico: dict) -> list[dict]:
        """Step 4: Flag anomalies in the results."""
        flags = []

        for name, res in results.items():
            if "error" in res:
                continue

            meta = res.get("meta", {})
            if "error" in meta:
                continue

            k = res.get("k", 0)

            # High heterogeneity
            i2_val = meta.get("i2", 0)
            if i2_val > I2_HIGH_THRESHOLD:
                action = "leave-one-out performed" if "leave_one_out" in res else "manual review needed"
                flags.append({
                    "type": "high_heterogeneity",
                    "outcome": name,
                    "description": f"I² = {i2_val:.1f}% exceeds {I2_HIGH_THRESHOLD}%",
                    "severity": "warning",
                    "action_taken": action,
                })

            # Weight dominance
            weights = meta.get("weights", [])
            if weights:
                max_weight = max(weights)
                if max_weight > WEIGHT_DOMINANCE_THRESHOLD:
                    flags.append({
                        "type": "weight_dominance",
                        "outcome": name,
                        "description": f"Single study carries {max_weight:.1%} of total weight",
                        "severity": "warning",
                        "action_taken": "influence analysis recommended",
                    })

            # Underpowered
            if k < MIN_K_WARNING:
                flags.append({
                    "type": "underpowered",
                    "outcome": name,
                    "description": f"Only k={k} studies — results may be unstable",
                    "severity": "info",
                    "action_taken": "noted in interpretation",
                })

            # Trim-and-fill direction flip
            taf = res.get("trim_and_fill", {})
            if taf.get("direction_flipped"):
                flags.append({
                    "type": "trim_fill_direction_flip",
                    "outcome": name,
                    "description": taf.get("warning", "Effect direction reversed after trim-and-fill"),
                    "severity": "critical",
                    "action_taken": "flagged for interpretation",
                })

            # Egger's test significant
            egger = res.get("egger", {})
            if egger.get("significant"):
                flags.append({
                    "type": "publication_bias_detected",
                    "outcome": name,
                    "description": f"Egger's test p={egger['p_value']:.4f} (significant at 0.10)",
                    "severity": "warning",
                    "action_taken": "publication bias noted",
                })

            # Prediction interval crosses null when CI doesn't
            pi = meta.get("prediction_interval")
            if pi and pi[0] is not None and pi[1] is not None:
                ci_excludes_null = (meta["ci_lower"] > 0 or meta["ci_upper"] < 0)
                pi_includes_null = (pi[0] <= 0 <= pi[1])
                if ci_excludes_null and pi_includes_null:
                    flags.append({
                        "type": "prediction_interval_warning",
                        "outcome": name,
                        "description": "CI excludes null but prediction interval includes it",
                        "severity": "info",
                        "action_taken": "noted — future studies may show different results",
                    })

        return flags

    def _interpret(self, results: dict, anomalies: list[dict],
                   pico: dict, quality: dict | None = None) -> dict:
        """Step 5: LLM interprets the statistical results."""
        user_content = (
            f"## PICO\n{json.dumps(pico, indent=2)}\n\n"
            f"## Statistical Results\n{json.dumps(results, default=str, indent=2)}\n\n"
            f"## Anomaly Flags\n{json.dumps(anomalies, indent=2)}\n\n"
        )
        if quality:
            user_content += f"## Quality Assessments\n{json.dumps(quality, default=str, indent=2)}\n\n"

        user_content += "Please interpret these results for use in the manuscript."

        messages = self._build_messages(
            user_content,
            system_override=self._load_round_prompt("statistician_interpret.yaml"),
        )

        try:
            response = self._call_llm(
                messages, response_format={"type": "json_object"},
                phase="statistician_interpret",
            )
            return self._parse_json(response, retry_messages=messages,
                                    phase="statistician_interpret")
        except LumenParseError:
            logger.warning("statistician_interpret_failed",
                           msg="Using placeholder interpretation")
            return {"interpretations": [], "overall_summary": "Interpretation unavailable",
                    "limitations": ["LLM interpretation failed"]}

    def _load_round_prompt(self, filename: str) -> str:
        """Load a specific prompt file."""
        from pathlib import Path
        import yaml as _yaml
        path = Path(__file__).resolve().parent.parent.parent / "prompts" / filename
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("system_prompt", data.get("prompt", ""))
        return ""
