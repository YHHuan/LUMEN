#!/usr/bin/env python3
"""
doi_hunter.py — 批次 DOI → PDF 下載工具
==========================================
瀑布流下載策略 (依序嘗試，成功即停):
  1. Unpaywall API      — 合法開放取用
  2. Europe PMC API     — PMC 開放取用
  3. Sci-Hub            — 影子圖書館 (domain 輪替)
  4. Library Genesis    — 影子圖書館 (scimag)

使用方式:
  1. 建立 dois.txt (每行一個 DOI) 或 dois.csv
  2. 設定 .env 或環境變數 UNPAYWALL_EMAIL
  3. python scripts/doi_hunter.py
  4. PDF 存入 ./pdfs/，報告見 ./report.csv

注意: Sci-Hub / LibGen 請確認在您的司法管轄區合法使用。
"""

import os
import csv
import sys
import time
import random
import logging
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple, List

import requests
from bs4 import BeautifulSoup

# ============================================================
# 基本設定
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 下載 PDF 存放目錄
OUTPUT_DIR = Path("./pdfs")
# 結果報告
REPORT_FILE = Path("./report.csv")
# HTTP 請求逾時 (秒)
TIMEOUT = 15
# 每次請求後的隨機延遲範圍 (秒)，避免被封鎖
JITTER_MIN = 1.0
JITTER_MAX = 3.0

# --- Sci-Hub 備用網域 ---
SCIHUB_DOMAINS: List[str] = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.mksa.top",
]

# --- Library Genesis 備用網域 ---
LIBGEN_DOMAINS: List[str] = [
    "http://libgen.is",
    "http://libgen.rs",
    "http://libgen.li",
]

# --- 備用 User-Agent 清單 (fake_useragent 不可用時的後備) ---
_FALLBACK_UAS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


# ============================================================
# 工具函數
# ============================================================

def get_user_agent() -> str:
    """取得隨機 User-Agent。優先使用 fake_useragent 套件，失敗則用備用清單。"""
    try:
        from fake_useragent import UserAgent
        return UserAgent(browsers=["chrome", "firefox", "safari"]).random
    except Exception:
        return random.choice(_FALLBACK_UAS)


def jitter():
    """加入隨機延遲，避免觸發速率限制或 IP 封鎖。"""
    time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))


def make_browser_headers(referer: str = "") -> dict:
    """建立仿真實瀏覽器的 HTTP Headers。"""
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def is_valid_pdf(path: Path) -> bool:
    """驗證檔案是否為合法 PDF (檢查 magic bytes 及最小檔案大小)。"""
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def download_pdf_from_url(url: str, output_path: Path) -> bool:
    """
    通用 PDF 下載函數。
    - 驗證 Content-Type (拒絕 HTML 頁面)
    - 驗證 PDF magic bytes (確保非假 PDF)
    - 失敗時自動清理殘留檔案
    """
    try:
        headers = {
            "User-Agent": get_user_agent(),
            "Accept": "application/pdf,*/*;q=0.8",
        }
        resp = requests.get(
            url, headers=headers, timeout=TIMEOUT,
            allow_redirects=True, stream=True,
        )
        if resp.status_code != 200:
            return False

        # 拒絕 Content-Type 為 HTML 的回應 (通常是錯誤頁面)
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type.lower() and "pdf" not in content_type.lower():
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if not is_valid_pdf(output_path):
            output_path.unlink(missing_ok=True)
            return False

        return True

    except Exception as e:
        logger.debug(f"下載失敗 ({url}): {e}")
        output_path.unlink(missing_ok=True) if output_path.exists() else None
        return False


def fix_url(url: str, base_domain: str) -> str:
    """將相對 URL 轉換為絕對 URL。處理 // 、/ 開頭及相對路徑。"""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    elif url.startswith("/"):
        return base_domain.rstrip("/") + url
    elif url.startswith("http"):
        return url
    else:
        return base_domain.rstrip("/") + "/" + url


# ============================================================
# 策略 1: Unpaywall API
# ============================================================

def try_unpaywall(doi: str, output_path: Path, email: str) -> bool:
    """
    透過 Unpaywall API 查詢開放取用 PDF。
    API 文件: https://unpaywall.org/products/api
    需要 email (免費，無需 API Key)。
    """
    if not email:
        logger.debug("未設定 email，跳過 Unpaywall")
        return False

    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        resp = requests.get(url, headers=make_browser_headers(), timeout=TIMEOUT)
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            logger.debug(f"Unpaywall → HTTP {resp.status_code}")
            return False

        data = resp.json()

        # 若非 Open Access，直接跳過
        if not data.get("is_oa"):
            logger.debug(f"Unpaywall: DOI {doi} 非 OA")
            return False

        # 嘗試最佳 OA 位置
        best_loc = data.get("best_oa_location") or {}
        pdf_url = best_loc.get("url_for_pdf") or best_loc.get("url")
        if pdf_url and download_pdf_from_url(pdf_url, output_path):
            return True

        # 遍歷所有 OA 位置
        for loc in data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf") or loc.get("url")
            if pdf_url and download_pdf_from_url(pdf_url, output_path):
                return True

    except Exception as e:
        logger.debug(f"Unpaywall 失敗 ({doi}): {e}")

    return False


# ============================================================
# 策略 2: Europe PMC
# ============================================================

def try_europepmc(doi: str, output_path: Path) -> bool:
    """
    透過 Europe PMC REST API 查詢 Open Access 全文 PDF。
    若有 PMCID，嘗試 Europe PMC 及 NCBI PMC 直接下載端點。
    """
    try:
        search_url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query=DOI:{urllib.parse.quote(doi, safe='')}"
            "&resultType=core&format=json&pageSize=1"
        )
        resp = requests.get(search_url, headers=make_browser_headers(), timeout=TIMEOUT)
        if resp.status_code != 200:
            return False

        results = resp.json().get("resultList", {}).get("result", [])
        if not results:
            return False

        article = results[0]
        pmcid = article.get("pmcid", "")
        is_oa = article.get("isOpenAccess", "N") == "Y"

        if not (is_oa and pmcid):
            return False

        # 端點 1: Europe PMC PDF render
        epmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        if download_pdf_from_url(epmc_url, output_path):
            return True

        # 端點 2: NCBI PMC PDF
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        if download_pdf_from_url(pmc_url, output_path):
            return True

    except Exception as e:
        logger.debug(f"Europe PMC 失敗 ({doi}): {e}")

    return False


# ============================================================
# 策略 3: Sci-Hub (影子圖書館)
# ============================================================

def _parse_scihub_pdf_url(soup: BeautifulSoup, domain: str) -> Optional[str]:
    """
    從 Sci-Hub 頁面解析真實 PDF 連結。
    優先順序:
      1. <iframe id="pdf" src="...">
      2. <embed src="...">
      3. <div id="pdf"> 中的任意連結
      4. 頁面中任何含 .pdf 的 <a href>
    """
    # 1. <iframe id="pdf">
    iframe = soup.find("iframe", id="pdf")
    if iframe and iframe.get("src"):
        return fix_url(iframe["src"], domain)

    # 2. <embed src="...">
    embed = soup.find("embed")
    if embed and embed.get("src"):
        src = embed["src"]
        if "pdf" in src.lower() or src.startswith(("/", "//")):
            return fix_url(src, domain)

    # 3. <div id="pdf"> 容器
    pdf_div = soup.find("div", id="pdf")
    if pdf_div:
        for tag in pdf_div.find_all(["iframe", "embed", "a"]):
            src = tag.get("src") or tag.get("href", "")
            if src:
                return fix_url(src, domain)

    # 4. 頁面中任何 .pdf 連結
    for a in soup.find_all("a", href=True):
        if ".pdf" in a["href"].lower():
            return fix_url(a["href"], domain)

    return None


def try_scihub(doi: str, output_path: Path) -> bool:
    """
    透過 Sci-Hub 下載 PDF。
    - 隨機打亂網域順序，分散請求負載
    - 每個網域失敗後繼續嘗試下一個
    注意: 請確認在您的司法管轄區使用 Sci-Hub 是合法的。
    """
    domains = SCIHUB_DOMAINS.copy()
    random.shuffle(domains)  # 隨機順序，避免固定 domain 被鎖定

    for domain in domains:
        try:
            page_url = f"{domain}/{doi}"
            headers = make_browser_headers(referer=domain)

            resp = requests.get(page_url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code != 200:
                logger.debug(f"Sci-Hub {domain} → HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            pdf_url = _parse_scihub_pdf_url(soup, domain)

            if not pdf_url:
                logger.debug(f"Sci-Hub {domain}: 找不到 PDF 連結")
                continue

            logger.debug(f"Sci-Hub 解析到 PDF URL: {pdf_url}")
            if download_pdf_from_url(pdf_url, output_path):
                return True

        except requests.exceptions.ConnectionError:
            logger.debug(f"Sci-Hub {domain}: 無法連線 (可能被封鎖)")
        except requests.exceptions.Timeout:
            logger.debug(f"Sci-Hub {domain}: 請求逾時")
        except Exception as e:
            logger.debug(f"Sci-Hub {domain} 未知錯誤: {e}")

        # 換 domain 前短暫等待
        time.sleep(random.uniform(0.5, 1.5))

    return False


# ============================================================
# 策略 4: Library Genesis scimag (影子圖書館)
# ============================================================

def _parse_libgen_links(soup: BeautifulSoup, domain: str) -> List[str]:
    """
    解析 LibGen scimag 搜尋結果頁面。
    LibGen 結果表格中的 [Download] 連結通常指向 library.lol 鏡像站。
    """
    links: List[str] = []

    # 優先解析 class="catalog" 表格 (LibGen 標準 HTML)
    table = soup.find("table", class_="catalog")
    if not table:
        table = soup.find("table")

    if table:
        for a in table.find_all("a", href=True):
            href = a["href"]
            # 辨識下載/鏡像連結
            if any(marker in href for marker in ["library.lol", "libgen.lc", "get.php", "ads.php"]):
                full = href if href.startswith("http") else domain.rstrip("/") + href
                links.append(full)

    # 備用方案: 搜尋整頁的 library.lol 連結
    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "library.lol" in href or "libgen.lc" in href:
                links.append(href)

    return links[:3]  # 最多嘗試 3 個，避免過多請求


def _follow_libgen_mirror(url: str, output_path: Path) -> bool:
    """
    追蹤 LibGen 鏡像頁面 (如 library.lol) 取得真實 PDF。
    library.lol 通常有 "GET" 按鈕連到實際 PDF 檔案。
    """
    try:
        headers = make_browser_headers(referer="http://libgen.is")
        resp = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        if resp.status_code != 200:
            return False

        # 若直接回傳 PDF
        content_type = resp.headers.get("content-type", "")
        if "application/pdf" in content_type:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            if is_valid_pdf(output_path):
                return True
            output_path.unlink(missing_ok=True)
            return False

        # 解析中間頁面，尋找最終下載連結
        soup = BeautifulSoup(resp.text, "html.parser")
        parsed_base = urllib.parse.urlparse(url)
        base = f"{parsed_base.scheme}://{parsed_base.netloc}"

        for a in soup.find_all("a", href=True):
            href = a["href"]
            link_text = a.get_text(strip=True).upper()
            # GET 按鈕、Download 文字或直接 .pdf 連結
            if (href.lower().endswith(".pdf")
                    or "get.php" in href
                    or link_text in ("GET", "DOWNLOAD", "TÉLÉCHARGER", "СКАЧАТЬ")):
                pdf_link = href if href.startswith("http") else base + href
                if download_pdf_from_url(pdf_link, output_path):
                    return True

    except Exception as e:
        logger.debug(f"LibGen 鏡像追蹤失敗 ({url}): {e}")

    return False


def try_libgen(doi: str, output_path: Path) -> bool:
    """
    透過 Library Genesis scimag 搜尋並下載 PDF。
    輪流嘗試多個備用網域，解析搜尋結果並追蹤鏡像連結。
    注意: 請確認在您的司法管轄區使用 LibGen 是合法的。
    """
    encoded_doi = urllib.parse.quote(doi, safe="")

    for domain in LIBGEN_DOMAINS:
        try:
            search_url = f"{domain}/scimag/?q={encoded_doi}"
            headers = make_browser_headers(referer=domain)

            resp = requests.get(search_url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
            if resp.status_code != 200:
                logger.debug(f"LibGen {domain} → HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            mirrors = _parse_libgen_links(soup, domain)

            if not mirrors:
                logger.debug(f"LibGen {domain}: 無搜尋結果 (DOI: {doi})")
                continue

            for mirror_url in mirrors:
                if _follow_libgen_mirror(mirror_url, output_path):
                    return True
                time.sleep(random.uniform(0.5, 1.0))

        except requests.exceptions.ConnectionError:
            logger.debug(f"LibGen {domain}: 無法連線")
        except requests.exceptions.Timeout:
            logger.debug(f"LibGen {domain}: 請求逾時")
        except Exception as e:
            logger.debug(f"LibGen {domain} 失敗: {e}")

        time.sleep(random.uniform(0.5, 1.5))

    return False


# ============================================================
# 主下載器 — 瀑布流策略調度器
# ============================================================

def download_doi(doi: str, output_dir: Path, email: str) -> Tuple[str, str]:
    """
    針對單一 DOI 依序嘗試所有下載策略。
    Returns: (status, source)
      status: "Success" | "Failed"
      source: "Unpaywall" | "EuropePMC" | "Sci-Hub" | "LibGen" | "already_exists" | "None"
    """
    # 將 DOI 中的特殊字元替換為底線，作為安全檔名
    safe_doi = doi.replace("/", "_").replace("\\", "_").replace(":", "_")
    output_path = output_dir / f"{safe_doi}.pdf"

    # 若已存在合法 PDF，直接跳過
    if is_valid_pdf(output_path):
        logger.info(f"  ⏭  已存在，跳過: {doi}")
        return "Success", "already_exists"

    # 定義策略清單: (來源名稱, callable)
    strategies = [
        ("Unpaywall",  lambda: try_unpaywall(doi, output_path, email)),
        ("EuropePMC",  lambda: try_europepmc(doi, output_path)),
        ("Sci-Hub",    lambda: try_scihub(doi, output_path)),
        ("LibGen",     lambda: try_libgen(doi, output_path)),
    ]

    for source_name, strategy in strategies:
        logger.info(f"  [{source_name}] 嘗試中...")
        try:
            if strategy():
                logger.info(f"  ✅ 成功 via {source_name}")
                return "Success", source_name
        except KeyboardInterrupt:
            raise  # 允許使用者中斷整個程式
        except Exception as e:
            logger.warning(f"  ⚠  {source_name} 發生例外: {e}")

        # 策略間加入隨機延遲
        jitter()

    logger.warning(f"  ❌ 所有策略均失敗")
    return "Failed", "None"


# ============================================================
# 輸入讀取
# ============================================================

def read_dois() -> List[str]:
    """
    從當前目錄讀取 DOI 清單。
    優先讀取 dois.txt (每行一個 DOI)，否則讀 dois.csv。
    自動跳過空行及 # 開頭的注釋行。
    """
    dois: List[str] = []

    if Path("dois.txt").exists():
        with open("dois.txt", "r", encoding="utf-8") as f:
            for line in f:
                doi = line.strip()
                if doi and not doi.startswith("#"):
                    dois.append(doi)
        logger.info(f"讀取 {len(dois)} 個 DOI (來源: dois.txt)")
        return dois

    if Path("dois.csv").exists():
        with open("dois.csv", "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # 自動偵測 DOI 欄位名稱 (大小寫不敏感)
            col_name = None
            for row in reader:
                if col_name is None:
                    for key in row.keys():
                        if "doi" in key.lower():
                            col_name = key
                            break
                    if col_name is None:
                        col_name = list(row.keys())[0]
                doi = row.get(col_name, "").strip()
                if doi:
                    dois.append(doi)
        logger.info(f"讀取 {len(dois)} 個 DOI (來源: dois.csv, 欄位: '{col_name}')")
        return dois

    raise FileNotFoundError(
        "找不到 dois.txt 或 dois.csv。\n"
        "請在當前目錄建立其中一個檔案後再執行。"
    )


# ============================================================
# 報告輸出
# ============================================================

def write_report(results: List[dict]):
    """將下載結果寫入 report.csv (UTF-8 BOM，Excel 相容)。"""
    with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["DOI", "Status", "Source"])
        writer.writeheader()
        writer.writerows(results)

    success = sum(1 for r in results if r["Status"] == "Success")
    logger.info("\n" + "=" * 55)
    logger.info(f"下載完成: {success} / {len(results)} 成功")
    logger.info(f"報告儲存至: {REPORT_FILE.resolve()}")
    logger.info("=" * 55)


# ============================================================
# 主程式
# ============================================================

def main():
    # 載入 .env 環境變數 (可選)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    email = os.getenv("UNPAYWALL_EMAIL", os.getenv("EMAIL", ""))
    if not email:
        logger.warning("⚠  未設定 UNPAYWALL_EMAIL — Unpaywall 策略將被跳過")
        logger.warning("   設定方式: export UNPAYWALL_EMAIL=you@example.com")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 讀取 DOI 清單
    try:
        dois = read_dois()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not dois:
        logger.warning("DOI 清單為空，程式結束。")
        return

    results: List[dict] = []
    total = len(dois)

    for i, doi in enumerate(dois, 1):
        logger.info(f"\n[{i:3d}/{total}] DOI: {doi}")
        try:
            status, source = download_doi(doi, OUTPUT_DIR, email)
        except KeyboardInterrupt:
            logger.info("\n⛔  使用者中斷，儲存目前進度...")
            results.append({"DOI": doi, "Status": "Interrupted", "Source": "N/A"})
            break
        results.append({"DOI": doi, "Status": status, "Source": source})

    write_report(results)


if __name__ == "__main__":
    main()
