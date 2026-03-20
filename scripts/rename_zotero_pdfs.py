#!/usr/bin/env python3
"""
Rename Zotero PDFs to match study IDs — v5
=============================================
  python scripts/rename_zotero_pdfs.py --source stage1            # Normal run
  python scripts/rename_zotero_pdfs.py --source stage1 --dry-run  # Preview
  python scripts/rename_zotero_pdfs.py --source stage1 --diagnose # Coverage report

v5 improvements:
  - PMID matching from filename/path
  - DOI matching with better normalization
  - Title extraction from "Author - Year - Title" Zotero format
  - First-N-words matching for truncated titles
  - Diagnostic mode shows missing PDFs
  - Reads stage2 included list by default
  - Filters out proceedings/abstract PDFs (bad PDF detection)
  - Minimum title length guard prevents short-string false matches
"""

import json, os, re, shutil, argparse, sys
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project import select_project, get_data_dir

# PDFs whose filenames indicate they are NOT primary study papers
_BAD_PDF_PATTERNS = [
    "epresentations", "paper abstract", "invited faculty",
    "mechanisms, clinical strategies", "error in author affiliations",
    "abstract supplement", "conference proceeding",
]

# Add Zotero filename substrings here to forcibly skip them (wrong modality, wrong paper, etc.)
# Key = substring to match against Zotero PDF filename (case-insensitive)
# Value = human-readable reason (for the log)
SKIP_FILES = {
    "CN&T_2024_Transcranial pulse stimulation": "TPS paper; RIS_cochrane_45 is a tDCS study",
}

STAGE1_JSON = STAGE2_JSON = ZOTERO_PDF_DIR = TARGET_DIR = ""


def _init_paths():
    global STAGE1_JSON, STAGE2_JSON, ZOTERO_PDF_DIR, TARGET_DIR
    dd = get_data_dir()
    STAGE1_JSON    = f"{dd}/phase3_screening/stage1_title_abstract/included_studies.json"
    STAGE2_JSON    = f"{dd}/phase3_screening/stage2_fulltext/included_studies.json"
    ZOTERO_PDF_DIR = f"{dd}/zotero_export"
    TARGET_DIR     = f"{dd}/phase3_screening/stage2_fulltext/fulltext_pdfs"


def normalize_text(text):
    if not text: return ""
    return re.sub(r'[\W_]+', ' ', text.lower()).strip()

def extract_title_from_zotero_name(filename):
    stem = filename.rsplit('.', 1)[0]
    parts = re.split(r'\s*[-–—]\s*', stem)
    if len(parts) >= 3:
        return parts[-1].strip()
    parts = stem.split('_', 2)
    if len(parts) >= 3:
        return parts[-1].strip()
    return stem

def extract_pmid_from_path(pdf_path):
    m = re.search(r'(?:PMID[_\s-]?)(\d{7,8})', str(pdf_path), re.IGNORECASE)
    return f"PMID_{m.group(1)}" if m else None

def is_bad_pdf(pdf_path):
    """Return True for known proceedings/abstract PDFs or explicit skip entries."""
    name_lower = pdf_path.name.lower()
    if any(pat in name_lower for pat in _BAD_PDF_PATTERNS):
        return True
    for key, reason in SKIP_FILES.items():
        if key.lower() in name_lower:
            return True
    return False

def safe_filename(study_id):
    return study_id.replace("/", "_").replace("\\", "_") + ".pdf"


def match_pdf_to_study(pdf_file, lookup_list):
    norm_pdf = normalize_text(pdf_file.stem)
    
    # Strategy 1: PMID in path
    pmid = extract_pmid_from_path(pdf_file)
    if pmid:
        for e in lookup_list:
            if e["study_id"] == pmid:
                return e, 1.0, "PMID Match"
    
    # Strategy 2: DOI in filename
    for e in lookup_list:
        if e["clean_doi"] and len(e["clean_doi"]) > 10 and e["clean_doi"] in norm_pdf:
            return e, 1.0, "DOI Match"
    
    # Strategy 3: Title matching
    title_portion = normalize_text(extract_title_from_zotero_name(pdf_file.name))
    best, score, reason = None, 0.0, ""
    
    for e in lookup_list:
        nt = e["norm_title"]
        if not nt: continue
        
        s1 = SequenceMatcher(None, nt, norm_pdf).ratio()
        s2 = SequenceMatcher(None, nt, title_portion).ratio() if title_portion != norm_pdf else 0
        # Containment only valid when the matched string is meaningfully long
        _MIN_LEN = 15
        s3 = (0.95 if (len(nt) >= _MIN_LEN and nt in norm_pdf) else
              0.90 if (title_portion and len(title_portion) >= _MIN_LEN and title_portion in nt) else 0)
        
        words = nt.split()[:6]
        s4 = 0.85 if len(words) >= 4 and ' '.join(words) in norm_pdf else 0
        
        s = max(s1, s2, s3, s4)
        if s > score:
            score, best = s, e
            reason = "Containment" if s3 > 0 else ("First-Words" if s4 > 0 else f"Similarity({s:.2f})")
    
    return best, score, reason


def diagnose_mode(studies, target_path):
    print("\n🔍 PDF Coverage Diagnostic")
    print("=" * 60)
    has_pdf = missing = 0
    missing_list = []
    for s in studies:
        safe_id = s["study_id"].replace("/", "_").replace("\\", "_")
        p = target_path / f"{safe_id}.pdf"
        if p.exists() and p.stat().st_size > 1000:
            has_pdf += 1
        else:
            missing += 1
            missing_list.append(s)
    
    print(f"  Total: {len(studies)} | Have PDF: {has_pdf} | Missing: {missing}")
    if missing_list:
        print(f"\n📋 Missing PDFs:")
        for s in missing_list:
            print(f"  ❌ {s['study_id']} | DOI: {s.get('doi','')} | {(s.get('title') or '')[:50]}")
    else:
        print("\n✅ All studies have PDFs!")
    return missing_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", choices=["stage1", "stage2"], default="stage2")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()
    select_project()
    _init_paths()

    json_source = STAGE1_JSON if args.source == "stage1" else STAGE2_JSON
    if not Path(json_source).exists():
        json_source = STAGE1_JSON
        print(f"⚠️ stage2 not found, using stage1")
    
    target_path = Path(TARGET_DIR)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 Loading: {json_source}")
    with open(json_source, 'r', encoding='utf-8') as f:
        studies = json.load(f)
    
    lookup = [{"study_id": s["study_id"],
               "norm_title": normalize_text(s.get("title", "")),
               "clean_doi": normalize_text(s.get("doi", ""))} for s in studies]
    
    if args.diagnose:
        diagnose_mode(studies, target_path)
        return
    
    source_path = Path(ZOTERO_PDF_DIR)
    if not source_path.exists():
        print(f"❌ Zotero folder not found: {source_path}")
        return
    
    pdfs = list(source_path.rglob("*.pdf"))
    print(f"📂 Found {len(pdfs)} PDFs in zotero_export\n")

    # Only try to match studies that still lack a PDF
    lookup_missing = [e for e in lookup
                      if not (target_path / safe_filename(e["study_id"])).exists()]
    print(f"🎯 Targeting {len(lookup_missing)} studies still missing PDFs "
          f"(skipping {len(lookup) - len(lookup_missing)} already covered)\n")

    matched = 0
    skipped_bad = 0
    for pdf in pdfs:
        if is_bad_pdf(pdf):
            skipped_bad += 1
            continue
        best, score, reason = match_pdf_to_study(pdf, lookup_missing)
        if best and score > args.threshold:
            dest = target_path / safe_filename(best["study_id"])
            if dest.exists():
                print(f"⚠️ [exists] {safe_filename(best['study_id'])}")
                continue
            if args.dry_run:
                print(f"🔍 {pdf.name[:40]}... → {safe_filename(best['study_id'])} [{reason}]")
            else:
                shutil.copy2(pdf, dest)
                print(f"✅ {pdf.name[:40]}... → {safe_filename(best['study_id'])} [{reason}]")
            matched += 1
        else:
            print(f"❌ No match: {pdf.name} (best={score:.2f})")
    
    print(f"\n🎉 Done: {matched} new PDFs matched/copied  "
          f"| {skipped_bad} bad/proceedings skipped")
    diagnose_mode(studies, target_path)

if __name__ == "__main__":
    main()
