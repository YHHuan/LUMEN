"""
File Handlers & Data Manager
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import logging

from src.utils.project import get_data_dir

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalar types to native Python types for JSON serialization."""
    def default(self, obj):
        if _NUMPY_AVAILABLE:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        return super().default(obj)

logger = logging.getLogger(__name__)


class DataManager:
    """統一管理所有階段的資料存取"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(get_data_dir())
    
    def save(self, phase: str, filename: str, data: Any, 
             subfolder: str = ""):
        """
        儲存資料到對應 phase 的目錄。
        
        Args:
            phase: "phase1_strategy", "phase2_search", etc.
            filename: 檔案名 (含副檔名)
            data: 要存的資料
            subfolder: 子目錄 (如 "raw", "deduplicated")
        """
        if subfolder:
            output_dir = self.base_dir / phase / subfolder
        else:
            output_dir = self.base_dir / phase
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        # Metadata tracking
        self._update_metadata(phase, subfolder, filename, data)
        logger.info(f"✅ Saved: {filepath}")
        return filepath
    
    def load(self, phase: str, filename: str, 
             subfolder: str = "") -> Any:
        """載入資料"""
        if subfolder:
            filepath = self.base_dir / phase / subfolder / filename
        else:
            filepath = self.base_dir / phase / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        
        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    
    def exists(self, phase: str, filename: str, 
               subfolder: str = "") -> bool:
        """檢查檔案是否存在"""
        if subfolder:
            filepath = self.base_dir / phase / subfolder / filename
        else:
            filepath = self.base_dir / phase / filename
        return filepath.exists()
    
    def list_files(self, phase: str, subfolder: str = "",
                   pattern: str = "*") -> list:
        """列出目錄下的檔案"""
        if subfolder:
            dir_path = self.base_dir / phase / subfolder
        else:
            dir_path = self.base_dir / phase
        
        if not dir_path.exists():
            return []
        return sorted(dir_path.glob(pattern))
    
    def _update_metadata(self, phase: str, subfolder: str, 
                         filename: str, data: Any):
        """追蹤每個檔案的 metadata"""
        meta_dir = self.base_dir / phase
        if subfolder:
            meta_dir = meta_dir / subfolder
        meta_file = meta_dir / "_metadata.json"
        
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        size = len(json.dumps(data, ensure_ascii=False, cls=_NumpyEncoder)) if not isinstance(data, str) else len(data)
        count = len(data) if isinstance(data, (list, dict)) else 1
        
        metadata[filename] = {
            "updated_at": datetime.now().isoformat(),
            "size_bytes": size,
            "record_count": count,
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
