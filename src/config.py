"""
Centralized Configuration Manager — v5
========================================
v5: 新增 Database Toggles (ENABLE_PUBMED, ENABLE_EUROPEPMC, etc.)

Usage:
    from src.config import cfg
    
    cfg.ncbi_api_key        # API key (str or None)
    cfg.has_scopus           # 是否有 Scopus key
    cfg.enable_pubmed        # 是否啟用 PubMed
    cfg.budget("phase3")     # 取得 phase 預算 (float)
    cfg.available_databases  # 可用且啟用的資料庫列表
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


class _Config:
    """Singleton config"""
    
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
    
    # ── Feature flags (key 有填 = 可用) ───────────────
    
    @property
    def has_pubmed(self) -> bool:
        return True
    
    @property
    def has_europepmc(self) -> bool:
        return True
    
    @property
    def has_scopus(self) -> bool:
        return bool(self.elsevier_api_key)
    
    @property
    def has_wos(self) -> bool:
        return bool(self.wos_api_key)
    
    @property
    def has_crossref(self) -> bool:
        return True
    
    @property
    def has_semantic_scholar(self) -> bool:
        return bool(self.semantic_scholar_api_key)
    
    @property
    def available_databases(self) -> list:
        """可用且啟用的資料庫"""
        dbs = []
        if self.enable_pubmed:
            dbs.append("pubmed")
        if self.enable_europepmc:
            dbs.append("europepmc")
        if self.enable_scopus and self.has_scopus:
            dbs.append("scopus")
        if self.has_wos:
            dbs.append("wos")
        if self.enable_crossref and self.has_crossref:
            dbs.append("crossref")
        if self.has_semantic_scholar:
            dbs.append("semantic_scholar")
        return dbs
    
    @property
    def skipped_databases(self) -> list:
        skipped = []
        if not self.enable_pubmed:
            skipped.append("pubmed (disabled)")
        if not self.enable_europepmc:
            skipped.append("europepmc (disabled)")
        if not self.has_scopus:
            skipped.append("scopus (no key)")
        elif not self.enable_scopus:
            skipped.append("scopus (disabled)")
        if not self.has_wos:
            skipped.append("wos (no key)")
        if not self.enable_crossref:
            skipped.append("crossref (disabled)")
        if not self.has_semantic_scholar:
            skipped.append("semantic_scholar (no key)")
        return skipped
    
    # ── Token Budgets ─────────────────────────────────
    
    def budget(self, phase: str) -> float:
        key = f"TOKEN_BUDGET_{phase.upper()}"
        return float(os.getenv(key, "10.0"))
    
    @property
    def budget_phase1(self) -> float:
        return float(os.getenv("TOKEN_BUDGET_PHASE1", "5.0"))
    
    @property
    def budget_phase3_ta(self) -> float:
        return float(os.getenv("TOKEN_BUDGET_PHASE3_TA", "10.0"))
    
    @property
    def budget_phase3_ft(self) -> float:
        return float(os.getenv("TOKEN_BUDGET_PHASE3_FT", "12.0"))
    
    @property
    def budget_phase4(self) -> float:
        return float(os.getenv("TOKEN_BUDGET_PHASE4", "15.0"))
    
    @property
    def budget_phase5(self) -> float:
        return float(os.getenv("TOKEN_BUDGET_PHASE5", "5.0"))
    
    @property
    def budget_phase6(self) -> float:
        return float(os.getenv("TOKEN_BUDGET_PHASE6", "15.0"))
    
    @property
    def budget_total(self) -> float:
        return float(os.getenv("TOKEN_BUDGET_TOTAL", "50.0"))
    
    def print_status(self):
        print("=" * 55)
        print("  Meta-Analysis Generator v5 — Config Status")
        print("=" * 55)
        
        def _mask(key: str) -> str:
            return f"✅ {key[:8]}..." if key else "❌ (empty)"
        
        print(f"\n  🤖 LLM")
        print(f"     OpenRouter:    {_mask(self.openrouter_api_key)}")
        
        print(f"\n  📚 Databases")
        print(f"     PubMed:        {_mask(self.ncbi_api_key)} {'[ON]' if self.enable_pubmed else '[OFF]'}")
        print(f"     Europe PMC:    ✅ (no key) {'[ON]' if self.enable_europepmc else '[OFF]'}")
        print(f"     Scopus:        {_mask(self.elsevier_api_key)} {'[ON]' if self.enable_scopus else '[OFF]'}")
        print(f"     Web of Science:{_mask(self.wos_api_key)}")
        print(f"     CrossRef:      ✅ (email: {self.crossref_email or '—'}) {'[ON]' if self.enable_crossref else '[OFF]'}")
        print(f"     Semantic Sch.: {_mask(self.semantic_scholar_api_key)}")
        
        print(f"\n  📥 PDF Download")
        print(f"     Unpaywall:     ✅ (email: {self.unpaywall_email or '—'})")
        
        print(f"\n  💰 Budgets (USD)")
        print(f"     Phase 1 Strategy:   ${self.budget_phase1}")
        print(f"     Phase 3 T/A Screen: ${self.budget_phase3_ta}")
        print(f"     Phase 3 Full-text:  ${self.budget_phase3_ft}")
        print(f"     Phase 4 Extraction: ${self.budget_phase4}")
        print(f"     Phase 5 Statistics: ${self.budget_phase5}")
        print(f"     Phase 6 Writing:    ${self.budget_phase6}")
        print(f"     TOTAL limit:        ${self.budget_total}")
        
        print(f"\n  Available: {', '.join(self.available_databases)}")
        if self.skipped_databases:
            print(f"  Skipped:   {', '.join(self.skipped_databases)}")
        print()


cfg = _Config()
