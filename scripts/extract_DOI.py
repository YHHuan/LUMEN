import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project import select_project

def main():
    project_dir = select_project()
    file_path = os.path.join(project_dir, "phase3_screening", "stage2_fulltext", "download_log.json")

    # 1. 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        print("請確認你的終端機 (Terminal) 是否在專案的根目錄下執行。")
        return

    # 2. 讀取 JSON 檔案
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"讀取 JSON 時發生錯誤: {e}")
        return

    # 3. 核心邏輯：提取 DOI
    valid_dois = []

    if isinstance(data, list):
        for entry in data:
            # 條件 1: 狀態必須是 manual_download_needed
            if entry.get("status") == "manual_download_needed":
                doi = entry.get("doi")
                
                # 條件 2: DOI 必須存在且不是空字串
                if doi and isinstance(doi, str) and doi.strip():
                    valid_dois.append(doi.strip())
    else:
        print("錯誤：JSON 內容格式不正確 (應為 List)")
        return

    # 去除重複
    unique_dois = list(set(valid_dois))

    # 4. 輸出結果
    print(f"成功讀取檔案: {file_path}")
    print(f"共找到 {len(unique_dois)} 筆需要下載的有效 DOI：\n")

    # 產生方便複製的字串 (換行分隔)
    output_text = "\n".join(unique_dois)

    print("=== 請複製下方內容 (可直接丟入 Zotero 魔術棒) ===")
    print(output_text)
    print("=================================================")

    # 存成文字檔
    output_filename = "dois_to_download.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n已將列表儲存為: {output_filename}")

if __name__ == "__main__":
    main()