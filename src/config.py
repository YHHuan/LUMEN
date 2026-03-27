"""
Centralized Configuration Manager — LUMEN v2
==============================================
Usage:
    from src.config import cfg

    cfg.openrouter_api_key
    cfg.available_databases
    cfg.budget("phase3_ta")
    cfg.v2  # v2_settings.yaml as dict
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


def _load_v2_settings() -> dict:
    path = _project_root / "config" / "v2_settings.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class _Config:
    """Singleton config — environment + v2_settings.yaml."""

    def __init__(self):
        self._v2 = _load_v2_settings()

    @property
    def v2(self) -> dict:
        return self._v2

    # ── API Keys ──────────────────────────────────────

    @property
    def openrouter_api_key(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def ncbi_api_key(self) -> str:
        return os.getenv("NCBI_API_KEY", "")

    @property
    def ncbi_email(self) -> str:
        return os.getenv("NCBI_EMAIL", "")

    @property
    def unpaywall_email(self) -> str:
        return os.getenv("UNPAYWALL_EMAIL", "")

    @property
    def crossref_email(self) -> str:
        return os.getenv("CROSSREF_EMAIL", "")

    @property
    def elsevier_api_key(self) -> str:
        return os.getenv("ELSEVIER_API_KEY", "")

    @property
    def elsevier_inst_token(self) -> str:
        return os.getenv("ELSEVIER_INST_TOKEN", "")

    @property
    def wos_api_key(self) -> str:
        return os.getenv("WOS_API_KEY", "")

    @property
    def semantic_scholar_api_key(self) -> str:
        return os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    # ── Database Toggles ──────────────────────────────

    @property
    def enable_pubmed(self) -> bool:
        return os.getenv("ENABLE_PUBMED", "true").lower() == "true"

    @property
    def enable_europepmc(self) -> bool:
        return os.getenv("ENABLE_EUROPEPMC", "true").lower() == "true"

    @property
    def enable_scopus(self) -> bool:
        return os.getenv("ENABLE_SCOPUS", "true").lower() == "true"

    @property
    def enable_crossref(self) -> bool:
        return os.getenv("ENABLE_CROSSREF", "true").lower() == "true"

    @property
    def enable_openalex(self) -> bool:
        return os.getenv("ENABLE_OPENALEX", "true").lower() == "true"

    @property
    def has_scopus(self) -> bool:
        return bool(self.elsevier_api_key)

    @property
    def has_wos(self) -> bool:
        return bool(self.wos_api_key)

    @property
    def has_semantic_scholar(self) -> bool:
        return bool(self.semantic_scholar_api_key)

    @property
    def available_databases(self) -> list:
        dbs = []
        if self.enable_pubmed:
            dbs.append("pubmed")
        if self.enable_europepmc:
            dbs.append("europepmc")
        if self.enable_scopus and self.has_scopus:
            dbs.append("scopus")
        if self.has_wos:
            dbs.append("wos")
        if self.enable_crossref:
            dbs.append("crossref")
        if self.enable_openalex:
            dbs.append("openalex")
        if self.has_semantic_scholar:
            dbs.append("semantic_scholar")
        return dbs

    # ── Token Budgets ─────────────────────────────────

    def budget(self, phase: str) -> float:
        key = f"TOKEN_BUDGET_{phase.upper()}"
        return float(os.getenv(key, "10.0"))

    # ── v2 settings shortcuts ─────────────────────────

    @property
    def extraction_settings(self) -> dict:
        return self._v2.get("extraction", {})

    @property
    def phase5_settings(self) -> dict:
        return self._v2.get("phase5", {})

    @property
    def phase6_settings(self) -> dict:
        return self._v2.get("phase6", {})

    @property
    def prescreen_settings(self) -> dict:
        return self._v2.get("pre_screening", {})

    @property
    def ablation_settings(self) -> dict:
        return self._v2.get("ablation", {})

    @property
    def quality_assessment_settings(self) -> dict:
        return self._v2.get("quality_assessment", {})

    @property
    def pdf_conversion_settings(self) -> dict:
        return self._v2.get("pdf_conversion", {})


cfg = _Config()
