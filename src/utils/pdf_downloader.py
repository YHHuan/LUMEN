"""
PDF Downloader & Text Extractor — v5
======================================
多來源下載策略 + 文字提取快取 + Front Matter 啟發式辨識。

下載順序:
1. Unpaywall (開放取用 PDF)
2. PubMed Central (PMC)
3. Europe PMC
4. Semantic Scholar
5. OpenAlex
6. Crossref
7. Direct DOI Resolution
8. Sci-Hub (domain rotation + iframe/embed parsing)
9. Library Genesis scimag (multi-domain + mirror redirect)
10. 標記為 "manual_download_needed" 並輸出 Proxy 救援清單
"""

import os
import csv
import time
import random
import logging
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple, List
import requests

# === 影子圖書館備用網域 ===
# 按可靠性排序；下載時會隨機化順序以分散負載
SCIHUB_DOMAINS: List[str] = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.mksa.top",
]

LIBGEN_DOMAINS: List[str] = [
    "http://libgen.is",
    "http://libgen.rs",
    "http://libgen.li",
]

# 模擬瀏覽器 User-Agent 備用清單 (當 fake_useragent 不可用時)
_FALLBACK_UAS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]

from src.utils.project import get_data_dir

logger = logging.getLogger(__name__)


class PDFDownloader:

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(get_data_dir()) / "phase3_screening" / "stage2_fulltext" / "fulltext_pdfs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        from src.apis.unpaywall import UnpaywallAPI
        self.unpaywall = UnpaywallAPI()
    
    def download(self, study: dict) -> Tuple[Optional[str], str]:
        """
        嘗試下載 PDF。
        Returns: (pdf_path or None, status)
        """
        study_id = study.get("study_id", "unknown")
        safe_id = study_id.replace("/", "_").replace("\\", "_")
        pdf_path = self.output_dir / f"{safe_id}.pdf"
        
        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            return str(pdf_path), "already_exists"
        
        doi = study.get("doi", "")
        pmc_id = study.get("pmc_id", "")
        
        # === Strategy 1: Unpaywall ===
        if doi:
            oa_info = self.unpaywall.find_open_access(doi)
            pdf_url = oa_info.get("pdf_url") or oa_info.get("best_oa_url")
            if pdf_url:
                if self._download_file(pdf_url, pdf_path):
                    logger.info(f"✅ Downloaded via Unpaywall: {study_id}")
                    return str(pdf_path), "downloaded"
        
        # === Strategy 2: PubMed Central ===
        if pmc_id:
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
            if self._download_file(pmc_url, pdf_path):
                logger.info(f"✅ Downloaded via PMC: {study_id}")
                return str(pdf_path), "downloaded"
        
        # === Strategy 3: Europe PMC ===
        if pmc_id:
            epmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc_id}&blobtype=pdf"
            if self._download_file(epmc_url, pdf_path):
                logger.info(f"✅ Downloaded via Europe PMC: {study_id}")
                return str(pdf_path), "downloaded"
        
        # === Strategy 4: Semantic Scholar ===
        if doi:
            s2_url = self._get_semantic_scholar_pdf(doi)
            if s2_url and self._download_file(s2_url, pdf_path):
                logger.info(f"✅ Downloaded via Semantic Scholar: {study_id}")
                return str(pdf_path), "downloaded"

        # === Strategy 5: OpenAlex ===
        if doi:
            oa_url = self._get_openalex_pdf(doi)
            if oa_url and self._download_file(oa_url, pdf_path):
                logger.info(f"✅ Downloaded via OpenAlex: {study_id}")
                return str(pdf_path), "downloaded"

        # === Strategy 6: Crossref ===
        if doi:
            cr_url = self._get_crossref_pdf(doi)
            if cr_url and self._download_file(cr_url, pdf_path):
                logger.info(f"✅ Downloaded via Crossref: {study_id}")
                return str(pdf_path), "downloaded"

        # === Strategy 7: Direct DOI ===
        if doi:
            if self._try_direct_doi_resolution(doi, pdf_path):
                logger.info(f"✅ Downloaded via Direct DOI: {study_id}")
                return str(pdf_path), "downloaded"

        # === Strategy 8: Sci-Hub ===
        if doi:
            if self._try_scihub(doi, pdf_path):
                logger.info(f"✅ Downloaded via Sci-Hub: {study_id}")
                return str(pdf_path), "downloaded"

        # === Strategy 9: Library Genesis ===
        if doi:
            if self._try_libgen(doi, pdf_path):
                logger.info(f"✅ Downloaded via LibGen: {study_id}")
                return str(pdf_path), "downloaded"

        logger.info(f"⚠️  Manual download needed: {study_id} (DOI: {doi})")
        return None, "manual_download_needed"
    
    def export_missing_to_csv(self, missing_studies: list, filename: str = "missing_pdfs_action_list.csv"):
        """匯出下載失敗清單 (含台大/交大 Proxy 連結 + 標題/作者，供無DOI文獻手動查找)"""
        if not missing_studies:
            logger.info("🎉 沒有需要手動下載的文獻！")
            return

        filepath = self.output_dir.parent / filename
        headers = ["study_id", "has_doi", "doi", "year", "authors", "title",
                   "ntu_proxy_link", "nycu_proxy_link", "direct_doi_link", "search_tip"]

        # If the target file is locked (e.g. open in Excel), fall back to a timestamped name
        try:
            filepath.open("a").close()
        except PermissionError:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filename.replace(".csv", f"_{ts}.csv")
            filepath = self.output_dir.parent / filename
            logger.warning(f"⚠️  Original CSV locked — writing to {filename} instead")

        try:
            with open(filepath, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for study in missing_studies:
                    doi = study.get("doi", "") or ""
                    year = study.get("year", "") or ""
                    authors = study.get("authors") or study.get("author") or ""
                    if isinstance(authors, list):
                        authors = "; ".join(str(a) for a in authors[:3])
                    title = study.get("title", "") or ""
                    if doi:
                        search_tip = f"https://doi.org/{doi}"
                    else:
                        search_tip = f'Search: "{title[:60]}" {year}'
                    writer.writerow({
                        "study_id": study.get("study_id", ""),
                        "has_doi": "YES" if doi else "NO",
                        "doi": doi,
                        "year": year,
                        "authors": str(authors)[:100],
                        "title": title,
                        "ntu_proxy_link": f"https://ezproxy.lib.ntu.edu.tw/login?url=https://doi.org/{doi}" if doi else "",
                        "nycu_proxy_link": f"https://ezproxy.lib.nycu.edu.tw/login?url=https://doi.org/{doi}" if doi else "",
                        "direct_doi_link": f"https://doi.org/{doi}" if doi else "",
                        "search_tip": search_tip,
                    })
            logger.info(f"📁 匯出 {len(missing_studies)} 篇至: {filepath}")
        except Exception as e:
            logger.error(f"❌ 匯出失敗: {e}")

    def _download_file(self, url: str, output_path: Path) -> bool:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/pdf,*/*;q=0.8",
            }
            resp = requests.get(url, headers=headers, timeout=60, allow_redirects=True, stream=True)
            if resp.status_code != 200:
                return False
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type.lower() and "pdf" not in content_type.lower():
                return False
            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            with open(output_path, 'rb') as f:
                if f.read(5) != b'%PDF-':
                    output_path.unlink()
                    return False
            return True
        except Exception as e:
            logger.debug(f"Download failed from {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _get_semantic_scholar_pdf(self, doi: str) -> Optional[str]:
        try:
            resp = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf",
                timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                oa = data.get("openAccessPdf")
                if oa and oa.get("url"):
                    return oa["url"]
        except Exception:
            pass
        return None

    def _get_openalex_pdf(self, doi: str) -> Optional[str]:
        from src.config import cfg
        email = cfg.unpaywall_email or "user@example.com"
        try:
            resp = requests.get(
                f"https://api.openalex.org/works/https://doi.org/{doi}?mailto={email}",
                timeout=10)
            if resp.status_code == 200:
                oa = resp.json().get("open_access", {})
                if oa.get("is_oa") and oa.get("oa_url"):
                    return oa["oa_url"]
        except Exception:
            pass
        return None

    def _get_crossref_pdf(self, doi: str) -> Optional[str]:
        try:
            resp = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
            if resp.status_code == 200:
                for link in resp.json().get("message", {}).get("link", []):
                    if link.get("content-type") == "application/pdf":
                        return link.get("URL")
        except Exception:
            pass
        return None

    def _try_direct_doi_resolution(self, doi: str, output_path: Path) -> bool:
        return self._download_file(f"https://doi.org/{doi}", output_path)

    # ------------------------------------------------------------------ #
    # Strategy 8: Sci-Hub                                                  #
    # ------------------------------------------------------------------ #

    def _try_scihub(self, doi: str, output_path: Path) -> bool:
        """
        透過 Sci-Hub 下載 PDF。
        - 隨機輪替多個備用網域以分散負載
        - 解析 <iframe id="pdf"> / <embed> / 任何 .pdf 連結
        - 修正 //domain 或 /path 相對 URL
        注意: 請確認在您的司法管轄區使用 Sci-Hub 是合法的。
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.debug("beautifulsoup4 未安裝，跳過 Sci-Hub 策略")
            return False

        domains = SCIHUB_DOMAINS.copy()
        random.shuffle(domains)  # 隨機順序，分散請求

        for domain in domains:
            try:
                page_url = f"{domain}/{doi}"
                headers = {
                    "User-Agent": random.choice(_FALLBACK_UAS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Referer": domain,
                }
                resp = requests.get(page_url, headers=headers, timeout=20, allow_redirects=True)
                if resp.status_code != 200:
                    logger.debug(f"Sci-Hub {domain} → HTTP {resp.status_code}")
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                pdf_url = self._parse_scihub_pdf_url(soup, domain)

                if not pdf_url:
                    logger.debug(f"Sci-Hub {domain}: 頁面中找不到 PDF 連結")
                    continue

                logger.debug(f"Sci-Hub 找到 PDF: {pdf_url}")
                if self._download_file(pdf_url, output_path):
                    return True

            except requests.exceptions.ConnectionError:
                logger.debug(f"Sci-Hub {domain}: 無法連線")
            except requests.exceptions.Timeout:
                logger.debug(f"Sci-Hub {domain}: 請求逾時")
            except Exception as e:
                logger.debug(f"Sci-Hub {domain} 失敗: {e}")

        return False

    @staticmethod
    def _parse_scihub_pdf_url(soup, domain: str) -> Optional[str]:
        """
        解析 Sci-Hub 頁面中的真實 PDF 連結。
        優先順序: <iframe id="pdf"> → <embed> → <div id="pdf"> → 任何 .pdf href
        """
        # 方法 1: <iframe id="pdf" src="...">
        iframe = soup.find("iframe", id="pdf")
        if iframe and iframe.get("src"):
            return PDFDownloader._fix_url(iframe["src"], domain)

        # 方法 2: <embed src="...">
        embed = soup.find("embed")
        if embed and embed.get("src"):
            src = embed["src"]
            if "pdf" in src.lower() or src.startswith(("/", "//")):
                return PDFDownloader._fix_url(src, domain)

        # 方法 3: <div id="pdf"> 容器中的任意連結
        pdf_div = soup.find("div", id="pdf")
        if pdf_div:
            for tag in pdf_div.find_all(["iframe", "embed", "a"]):
                src = tag.get("src") or tag.get("href", "")
                if src:
                    return PDFDownloader._fix_url(src, domain)

        # 方法 4: 頁面中任何含 .pdf 的 <a href>
        for a in soup.find_all("a", href=True):
            if ".pdf" in a["href"].lower():
                return PDFDownloader._fix_url(a["href"], domain)

        return None

    @staticmethod
    def _fix_url(url: str, base_domain: str) -> str:
        """修正相對路徑 URL 為絕對 URL"""
        url = url.strip()
        if url.startswith("//"):
            return "https:" + url
        elif url.startswith("/"):
            return base_domain.rstrip("/") + url
        elif url.startswith("http"):
            return url
        else:
            return base_domain.rstrip("/") + "/" + url

    # ------------------------------------------------------------------ #
    # Strategy 9: Library Genesis (scimag)                                 #
    # ------------------------------------------------------------------ #

    def _try_libgen(self, doi: str, output_path: Path) -> bool:
        """
        透過 Library Genesis scimag 搜尋並下載 PDF。
        - 嘗試多個備用網域 (libgen.is / .rs / .li)
        - 解析搜尋結果表格，取得鏡像下載連結
        - 追蹤 library.lol 等中間頁面取得最終 PDF
        注意: 請確認在您的司法管轄區使用 LibGen 是合法的。
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.debug("beautifulsoup4 未安裝，跳過 LibGen 策略")
            return False

        encoded_doi = urllib.parse.quote(doi, safe="")

        for domain in LIBGEN_DOMAINS:
            try:
                search_url = f"{domain}/scimag/?q={encoded_doi}"
                headers = {
                    "User-Agent": random.choice(_FALLBACK_UAS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Referer": domain,
                }
                resp = requests.get(search_url, headers=headers, timeout=20, allow_redirects=True)
                if resp.status_code != 200:
                    logger.debug(f"LibGen {domain} → HTTP {resp.status_code}")
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                download_links = self._parse_libgen_links(soup, domain)

                if not download_links:
                    logger.debug(f"LibGen {domain}: 找不到下載連結 (DOI: {doi})")
                    continue

                for link in download_links:
                    if self._follow_libgen_and_download(link, output_path):
                        return True
                    time.sleep(random.uniform(0.5, 1.5))

            except requests.exceptions.ConnectionError:
                logger.debug(f"LibGen {domain}: 無法連線")
            except requests.exceptions.Timeout:
                logger.debug(f"LibGen {domain}: 請求逾時")
            except Exception as e:
                logger.debug(f"LibGen {domain} 失敗: {e}")

        return False

    @staticmethod
    def _parse_libgen_links(soup, domain: str) -> List[str]:
        """
        解析 LibGen scimag 搜尋結果頁面中的下載連結。
        LibGen 搜尋結果表格中每筆有 [Download] 連結指向 library.lol 等鏡像站。
        """
        links: List[str] = []

        # 優先解析 .catalog 表格 (LibGen 標準樣式)
        table = soup.find("table", class_="catalog")
        if not table:
            table = soup.find("table")

        if table:
            for a in table.find_all("a", href=True):
                href = a["href"]
                if any(x in href for x in ["library.lol", "libgen.lc", "get.php", "ads.php"]):
                    full = href if href.startswith("http") else domain.rstrip("/") + href
                    links.append(full)

        # 備用: 頁面中所有 library.lol / libgen.lc 連結
        if not links:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "library.lol" in href or "libgen.lc" in href:
                    links.append(href)

        return links[:3]  # 最多嘗試 3 個鏡像以節省時間

    def _follow_libgen_and_download(self, url: str, output_path: Path) -> bool:
        """
        追蹤 LibGen 鏡像連結直到取得 PDF。
        library.lol 通常有一個中間頁面含 "GET" 按鈕指向真實 PDF。
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return False

        try:
            headers = {
                "User-Agent": random.choice(_FALLBACK_UAS),
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                "Referer": "http://libgen.is",
            }
            resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            if resp.status_code != 200:
                return False

            # 若直接回傳 PDF，寫入並驗證
            content_type = resp.headers.get("content-type", "")
            if "application/pdf" in content_type:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                with open(output_path, "rb") as f:
                    if f.read(5) == b"%PDF-":
                        return True
                output_path.unlink(missing_ok=True)
                return False

            # 解析中間下載頁 (library.lol 等)
            soup = BeautifulSoup(resp.text, "html.parser")
            parsed_base = urllib.parse.urlparse(url)
            base = f"{parsed_base.scheme}://{parsed_base.netloc}"

            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True).upper()
                # 找 GET / Download 按鈕 或 .pdf 連結
                if href.endswith(".pdf") or "get.php" in href or text in ("GET", "DOWNLOAD", "TÉLÉCHARGER"):
                    pdf_link = href if href.startswith("http") else base + href
                    if self._download_file(pdf_link, output_path):
                        return True

        except Exception as e:
            logger.debug(f"LibGen 追蹤失敗 ({url}): {e}")

        return False


class PDFTextExtractor:
    """PDF 文字提取 + section 辨識 + 快取"""
    
    def __init__(self):
        from src.utils.cache import PDFTextCache
        self.cache = PDFTextCache()
    
    def extract(self, pdf_path: str, max_tokens: int = 15000) -> dict:
        cached = self.cache.get(pdf_path, max_tokens)
        if cached is not None:
            logger.info(f"PDF cache hit ({max_tokens} tok limit): {pdf_path}")
            return cached
        result = self._extract_with_fitz(pdf_path)
        if result["tokens_approx"] > max_tokens:
            result = self._smart_truncate(result, max_tokens)
        self.cache.set(pdf_path, result, max_tokens)
        return result
    
    def _extract_with_fitz(self, pdf_path: str) -> dict:
        import fitz
        doc = fitz.open(pdf_path)
        full_text = ""
        page_texts = {}
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            page_texts[page_num] = page_text
            full_text += f"\n[PAGE {page_num}]\n{page_text}"
        pages = len(doc)
        doc.close()
        tables = self._extract_tables_pdfplumber(pdf_path)
        sections = self._identify_sections(full_text)
        return {
            "full_text": full_text,
            "sections": sections,
            "page_texts": page_texts,
            "tables": tables,
            "pages": pages,
            "tokens_approx": len(full_text) // 4,
        }
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> list:
        try:
            import pdfplumber
        except ImportError:
            return []
        tables_found = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    for t_idx, table in enumerate(page.extract_tables()):
                        if not table or len(table) < 2:
                            continue
                        headers = [str(h).strip() if h else "" for h in table[0]]
                        rows = [[str(c).strip() if c else "" for c in row] for row in table[1:]]
                        md_lines = ["| " + " | ".join(headers) + " |"]
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for row in rows:
                            padded = row + [""] * (len(headers) - len(row))
                            md_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
                        tables_found.append({
                            "page": page_num, "table_index": t_idx,
                            "headers": headers, "rows": rows,
                            "markdown": "\n".join(md_lines),
                            "n_rows": len(rows), "n_cols": len(headers),
                        })
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        return tables_found
    
    def _identify_sections(self, text: str) -> dict:
        import re
        patterns = [
            ("abstract", r'(?i)\b(abstract)\b'),
            ("introduction", r'(?i)\b(introduction|background)\b'),
            ("methods", r'(?i)\b(methods?|materials?\s+and\s+methods?|study\s+design|participants)\b'),
            ("results", r'(?i)\b(results?)\b'),
            ("discussion", r'(?i)\b(discussion)\b'),
            ("conclusion", r'(?i)\b(conclusions?)\b'),
            ("references", r'(?i)\b(references?|bibliography)\b'),
        ]
        positions = []
        for name, pattern in patterns:
            for match in re.finditer(pattern, text):
                pos = match.start()
                line_start = text.rfind('\n', 0, pos) + 1
                line = text[line_start:pos + len(match.group())].strip()
                if len(line) < 60:
                    positions.append((pos, name))
        positions.sort(key=lambda x: x[0])
        sections = {}
        if positions and positions[0][0] > 0:
            front = text[:positions[0][0]].strip()
            if len(front) > 200:
                sections['front_matter'] = front
        for i, (pos, name) in enumerate(positions):
            if name in sections:
                continue
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            section_text = text[pos:end].strip()
            lines = section_text.split('\n', 1)
            sections[name] = lines[1].strip() if len(lines) > 1 else section_text
        return sections
    
    def _smart_truncate(self, result: dict, max_tokens: int) -> dict:
        sections = result.get("sections", {})
        priority = ["methods", "results", "abstract", "front_matter", "conclusion"]
        truncated_text = ""
        for name in priority:
            if name in sections:
                truncated_text += f"\n\n=== {name.upper()} ===\n{sections[name]}"
        current_tokens = len(truncated_text) // 4
        if current_tokens > max_tokens:
            ratio = max_tokens / current_tokens
            for name in priority:
                if name in sections:
                    sections[name] = sections[name][:int(len(sections[name]) * ratio)] + "\n[... truncated ...]"
            truncated_text = ""
            for name in priority:
                if name in sections:
                    truncated_text += f"\n\n=== {name.upper()} ===\n{sections[name]}"
        return {
            "full_text": truncated_text,
            "sections": {k: v for k, v in sections.items() if k in priority},
            "tables": result.get("tables", []),  # preserve tables for Phase 4
            "pages": result["pages"],
            "tokens_approx": len(truncated_text) // 4,
            "truncated": True,
        }
    
    def extract_for_screening(self, pdf_path: str) -> str:
        result = self.extract(pdf_path, max_tokens=5000)
        sections = result.get("sections", {})
        text = ""
        if "abstract" in sections:
            text += f"ABSTRACT:\n{sections['abstract'][:2000]}\n\n"
        elif "front_matter" in sections:
            text += f"FRONT MATTER:\n{sections['front_matter'][:2500]}\n\n"
        if "methods" in sections:
            text += f"METHODS:\n{sections['methods'][:3000]}\n\n"
        if "results" in sections:
            text += f"RESULTS:\n{sections['results'][:2000]}\n\n"
        return text or result["full_text"][:5000]
    
    def extract_for_data_extraction(self, pdf_path: str) -> tuple:
        """Returns (full_text: str, tables: list) for Phase 4 extraction.
        Uses a 12000-token limit cached separately from the 5000-token
        screening result."""
        result = self.extract(pdf_path, max_tokens=12000)
        return result["full_text"], result.get("tables", [])
