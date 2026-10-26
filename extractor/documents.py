"""파서 출력(JSON) → LawDocument 로드."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LawDocument:
    law_id: str
    title: str
    short_title: Optional[str]
    doc_type: str
    doc_type_raw: str = ""
    issuing_org: Optional[str] = None
    promulgation_no: Optional[str] = None
    promulgation_date: Optional[str] = None
    effective_date: Optional[str] = None
    revision_type: Optional[str] = None
    jurisdiction: List[str] = field(default_factory=list)
    articles: List[Dict[str, Any]] = field(default_factory=list)
    addenda: List[Dict[str, Any]] = field(default_factory=list)


def load_documents(raw_dir: str | Path) -> List[LawDocument]:
    docs: List[LawDocument] = []
    for p in sorted(Path(raw_dir).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d.pop("raw_text", None)
        docs.append(LawDocument(**d))
    return docs
