"""엔티티(법령명) 정규화 — 표기 변이 · 약칭 · 문서 내 정의어를 정식 명칭으로 통합.

같은 법령이 KG 안에서 여러 노드로 갈라지면 임베딩 학습 신호가 분산된다.
v1.0 은 수작업 사전(SYNONYM_MAP) 7개 항목만 사용했고, v1.1 은 아래 세 가지를 자동으로 처리한다.

1. 표기 변이   : PDF 추출 공백 오류("방위사업법 시 행규칙"), 가운뎃점 변이(·/ㆍ/・)
                → 공백·가운뎃점을 지운 키가 같으면 같은 법령으로 보고 대표 표기로 통일
2. 약칭        : 법령 메타데이터의 약칭(예: 국방과학기술혁신법 → 국방과학기술혁신 촉진법)
                + 시행령/시행규칙 접미 변형
3. 문서 내 정의 : 「국방과학기술혁신 촉진법」(이하 "혁신법"이라 한다) 같은 정의문을 읽어
                해당 문서 안에서만 "혁신법" → 정식 명칭으로 해석
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

from .documents import LawDocument

# v1.0 수작업 사전 (논문 데이터셋 재현용)
SYNONYM_MAP: Dict[str, str] = {
    "국가계약법": "국가를 당사자로 하는 계약에 관한 법률",
    "지방계약법": "지방자치단체를 당사자로 하는 계약에 관한 법률",
    "대외무역법": "대외무역법",
    "공직자윤리법": "공직자윤리법",
    "방위사업법": "방위사업법",
    "국가계약법 시행령": "국가를 당사자로 하는 계약에 관한 법률 시행령",
    "국가계약법 시행규칙": "국가를 당사자로 하는 계약에 관한 법률 시행규칙",
}


def canonical_name_v10(raw_name: str) -> str:
    raw_name = raw_name.strip()
    return SYNONYM_MAP.get(raw_name, raw_name)


_DOTS = re.compile(r"[·ㆍ・•‧]")
_WS = re.compile(r"\s+")
SUFFIXES = ["", " 시행령", " 시행규칙", " 시행세칙"]
DEF_RE = re.compile(
    # 정의 괄호가 「법령명」 바로 뒤에 붙는 경우만 인정한다.
    # (「방위사업법」 제32조에 따른 국방기술품질원(이하 "기술품질원"…) 처럼 다른 대상을 정의하는 문장 배제)
    r"「\s*([^」]+?)\s*」\s*"
    r"\(\s*이하\s*[\"“”']([^\"“”']+)[\"“”']\s*(?:이?라|으로)\s*한다\s*\)"
)


def surface_key(name: str) -> str:
    """표기 비교용 키 : 공백 제거 + 가운뎃점 통일."""
    return _WS.sub("", _DOTS.sub("·", name.strip()))


class EntityNormalizer:
    def __init__(self, documents: List[LawDocument], all_mentions: Iterable[str] = ()):
        self.titles = {d.title for d in documents}
        # 1) 표기 변이 : 키 → 대표 표기
        counts = Counter(m.strip() for m in all_mentions)
        by_key: Dict[str, List[str]] = defaultdict(list)
        for name in set(counts) | self.titles:
            by_key[surface_key(name)].append(name)
        self.key2canon: Dict[str, str] = {}
        for key, names in by_key.items():
            in_corpus = [n for n in names if n in self.titles]
            if in_corpus:
                canon = in_corpus[0]
            else:  # 가장 많이 쓰인 표기, 동률이면 공백이 많은(=띄어쓰기 원형에 가까운) 표기
                canon = max(names, key=lambda n: (counts[n], n.count(" "), -len(n)))
            self.key2canon[key] = canon
        # 2) 약칭 (메타데이터 short_title) — 시행령/시행규칙 변형 포함
        self.abbrev: Dict[str, str] = {}
        for d in documents:
            if d.short_title:
                base_short, base_full = d.short_title, d.title
                for suf in [" 시행령", " 시행규칙", " 시행세칙"]:
                    if base_short.endswith(suf) and base_full.endswith(suf):
                        base_short, base_full = base_short[: -len(suf)], base_full[: -len(suf)]
                        break
                for suf in SUFFIXES:
                    self.abbrev[surface_key(base_short + suf)] = base_full + suf
        # 3) 문서 내 정의어 : law_id → {alias_key: target}
        self.local: Dict[str, Dict[str, str]] = {}
        for d in documents:
            amap: Dict[str, str] = {}
            for art in d.articles:
                for m in DEF_RE.finditer(art.get("text", "")):
                    target = " ".join(m.group(1).split())
                    alias = " ".join(m.group(2).split())
                    if alias == target or len(alias) < 2:
                        continue
                    for suf in SUFFIXES:
                        amap.setdefault(surface_key(alias + suf), target + suf)
            if amap:
                self.local[d.law_id] = amap
        self.stats = Counter()

    def resolve(self, name: str, doc: Optional[LawDocument] = None) -> str:
        name = name.strip()
        key = surface_key(name)
        if doc is not None and key in self.local.get(doc.law_id, {}):
            out, why = self.local[doc.law_id][key], "local_alias"
        elif key in self.abbrev:
            out, why = self.abbrev[key], "short_title"
        elif name in SYNONYM_MAP:
            out, why = SYNONYM_MAP[name], "synonym_map"
        else:
            out, why = name, "surface"
        out = self.key2canon.get(surface_key(out), out)   # 표기 변이 통일
        if out != name:
            self.stats[why] += 1
        return out
