"""스키마(JSON) 로드 + 클래스 계층 질의."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class Schema:
    namespace: str
    classes: Dict[str, dict]
    properties: Dict[str, dict]
    asymmetric_hint: Set[str] = field(default_factory=set)
    type_relation: str = "type"

    # ── 클래스 계층 ──
    def parent(self, c: str) -> Optional[str]:
        return (self.classes.get(c) or {}).get("parent")

    def ancestors(self, c: str) -> List[str]:
        out, cur = [], self.parent(c)
        while cur and cur not in out:
            out.append(cur)
            cur = self.parent(cur)
        return out

    def is_subclass(self, c: str, sup: str) -> bool:
        return c == sup or sup in self.ancestors(c)

    def depth(self, c: str) -> int:
        return len(self.ancestors(c))

    # ── 속성 ──
    def has_char(self, rel: str, char: str) -> bool:
        return char in self.properties.get(rel, {}).get("characteristics", [])

    def inverse(self, rel: str) -> Optional[str]:
        return self.properties.get(rel, {}).get("inverse")

    def iri(self, name: str) -> str:
        p = self.properties.get(name)
        return self.namespace + (p["iri"] if p else name)


def load_schema(path: str | Path) -> Schema:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Schema(namespace=d["namespace"], classes=d["classes"], properties=d["properties"],
                  asymmetric_hint=set(d.get("asymmetric_hint", [])),
                  type_relation=d.get("type_relation", "type"))
