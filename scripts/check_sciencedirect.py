#!/usr/bin/env python3
"""
ScienceDirect / Elsevier Access Diagnostic
============================================
Checks whether your IP can access ScienceDirect and the Elsevier API.

  python scripts/check_sciencedirect.py

Exit codes:
  0 — access OK
  1 — blocked or restricted
"""

import sys
import json
import requests

# A well-known open-access article with a ScienceDirect landing page
TEST_DOI      = "10.1016/j.neuron.2015.06.014"
SD_ARTICLE_URL = f"https://www.sciencedirect.com/science/article/pii/S0896627315005486"
DOI_RESOLVE    = f"https://doi.org/{TEST_DOI}"
ELSEVIER_ABSTRACT_URL = f"https://api.elsevier.com/content/abstract/doi/{TEST_DOI}"

HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}


def check(label: str, url: str, headers: dict = None, timeout: int = 15) -> tuple[int, str]:
    try:
        resp = requests.get(url, headers=headers or {}, timeout=timeout,
                            allow_redirects=True)
        return resp.status_code, resp.url
    except requests.exceptions.ConnectionError as e:
        return -1, f"ConnectionError: {e}"
    except requests.exceptions.Timeout:
        return -2, "Timeout"
    except Exception as e:
        return -3, str(e)


def main():
    print("=" * 60)
    print("ScienceDirect / Elsevier Access Diagnostic")
    print("=" * 60)

    issues = []

    # 1. Direct ScienceDirect article page
    code, final_url = check("ScienceDirect article page", SD_ARTICLE_URL, HEADERS_BROWSER)
    print(f"\n1. ScienceDirect article page")
    print(f"   URL: {SD_ARTICLE_URL}")
    print(f"   Status: {code}  →  Final URL: {final_url[:100]}")
    if code == 200:
        print("   ✅ Page reached successfully")
    elif code == 403:
        print("   ❌ 403 Forbidden — your IP may be blocked by Elsevier")
        issues.append("ScienceDirect 403")
    elif code == 429:
        print("   ⚠️  429 Too Many Requests — rate limited")
        issues.append("ScienceDirect 429")
    elif code < 0:
        print(f"   ❌ Network error: {final_url}")
        issues.append(f"ScienceDirect network error")
    else:
        print(f"   ⚠️  Unexpected status {code}")
        if "sciencedirect.com/user/login" in final_url or "login" in final_url.lower():
            print("   ⚠️  Redirected to login page — institutional access required")
            issues.append("ScienceDirect redirected to login")

    # 2. DOI resolution (should redirect to ScienceDirect for Elsevier papers)
    code2, final2 = check("DOI resolution", DOI_RESOLVE, HEADERS_BROWSER)
    print(f"\n2. DOI resolution → {DOI_RESOLVE}")
    print(f"   Status: {code2}  →  Final URL: {final2[:100]}")
    if code2 == 200 and "sciencedirect" in final2:
        print("   ✅ DOI resolves to ScienceDirect")
    elif code2 == 200:
        print(f"   ✅ DOI resolved (but not to ScienceDirect: {final2[:60]})")
    else:
        print(f"   ⚠️  Status {code2}")

    # 3. Elsevier Abstract API (requires API key)
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("ELSEVIER_API_KEY", "")

    print(f"\n3. Elsevier Abstract Retrieval API")
    if not api_key:
        print("   ⏸️  ELSEVIER_API_KEY not set — skipping API test")
    else:
        code3, final3 = check(
            "Elsevier API",
            ELSEVIER_ABSTRACT_URL,
            headers={"X-ELS-APIKey": api_key, "Accept": "application/json"},
        )
        print(f"   URL: {ELSEVIER_ABSTRACT_URL}")
        print(f"   Status: {code3}  →  {final3[:80]}")
        if code3 == 200:
            print("   ✅ API accessible with your key")
        elif code3 == 401:
            print("   ❌ 401 Unauthorized — invalid API key or IP not allowlisted")
            issues.append("Elsevier API 401")
        elif code3 == 403:
            print("   ❌ 403 Forbidden — API key valid but no entitlement for this endpoint")
            issues.append("Elsevier API 403")
        elif code3 == 429:
            print("   ⚠️  429 Too Many Requests — API quota exceeded")
            issues.append("Elsevier API 429")
        else:
            print(f"   ⚠️  Status {code3}")

    # 4. Unpaywall check (free, open-access meta)
    unpaywall_url = f"https://api.unpaywall.org/v2/{TEST_DOI}?email=test@example.com"
    code4, _ = check("Unpaywall", unpaywall_url)
    print(f"\n4. Unpaywall (open access metadata)")
    print(f"   Status: {code4}")
    if code4 == 200:
        print("   ✅ Unpaywall accessible — OA PDFs can be fetched without ScienceDirect")
    else:
        print(f"   ⚠️  Status {code4}")

    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("✅ No access issues detected.")
        print("   If PDF downloads from ScienceDirect are failing, the papers")
        print("   may be paywalled. Use institutional VPN/proxy for access,")
        print("   or rely on Unpaywall / Sci-Hub as fallback sources.")
    else:
        print(f"❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nRecommendations:")
        if any("403" in i or "blocked" in i.lower() for i in issues):
            print("  • Your IP may be blocked by Elsevier's bot-detection.")
            print("  • Use your institutional VPN or proxy.")
            print("  • The pipeline will use Sci-Hub / LibGen as fallback for PDFs.")
        if any("login" in i.lower() for i in issues):
            print("  • Connect to institutional network/VPN for ScienceDirect access.")
        print("\nNote: Search results are unaffected — Scopus API is separate from")
        print("  ScienceDirect and uses an API key (not browser access).")
    print("=" * 60)

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
