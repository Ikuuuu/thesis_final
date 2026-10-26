"""규칙 기반 관계 추출 — 인용 · 위임 · 상하위 법령 계층.

법령 문서는 형식이 매우 정형화되어 있어(「법령명」 제N조, "…에서 위임한 사항", 시행령/시행규칙 명명 규칙)
LLM 없이도 높은 정밀도로 관계를 뽑을 수 있다. 본 모듈이 졸업논문 KG(v1.0)를 만든 추출기이며,
`legacy=False`(v1.1)에서는 가지조문 인용(제N조의M) 인식을 보강했다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from .documents import LawDocument

# v1.0 : "제5조의2" 를 "제5의2조" 로 가정 → 가지조문 인용이 "제5조" 로 잘못 연결됨
CITATION_RE_V10 = re.compile(r"「\s*([^」]+?)\s*」(?:\s*제(\d+(?:의\d+)?)\s*조)?")
# v1.1 : 법령 표기 "제5조의2" 를 올바르게 인식
CITATION_RE = re.compile(r"「\s*([^」]+?)\s*」(?:\s*제(\d+)\s*조(?:\s*의\s*(\d+))?)?")

# 법령 위계 (숫자가 작을수록 상위). v1.1 에서 위임·상위법령 관계의 방향 검증에 사용한다.
DOC_RANK = {
    "Law": 0, "EnforcementOrdinance": 1, "SpecialRegulation": 1, "EnforcementRule": 2,
    "AdministrativeOrder": 3, "JointAdministrativeOrder": 3,
    "AdministrativeRule": 3, "AdministrativeNotice": 3,
}


def rank_ok(parent_doc: LawDocument, child_doc: LawDocument) -> bool:
    """상위 문서의 위계가 하위 문서보다 엄격히 높을 때만 True (온톨로지 기반 방향 검증)."""
    return DOC_RANK.get(parent_doc.doc_type, 9) < DOC_RANK.get(child_doc.doc_type, 9)


DELEGATE_KEYWORDS = [
    "위임된 사항", "위임받은 사항", "위임에 따라",
    "위임에 의하여", "위임에 의한", "에서 위임한", "에 따라",
    "에 근거하여", "규정에 의하여", "에서 위임된",
]


def _iter_citations(text: str, legacy: bool):
    if legacy:
        for m in CITATION_RE_V10.finditer(text):
            yield m.group(1).strip(), (m.group(2) or "")
    else:
        for m in CITATION_RE.finditer(text):
            no = m.group(2) or ""
            if no and m.group(3):
                no = f"{no}의{m.group(3)}"
            yield m.group(1).strip(), no


def extract_citations_from_article(article_text: str, self_title: str,
                                   legacy: bool = False) -> List[Dict[str, str]]:
    """조문 본문의 「법령명」 (제N조) 인용을 추출한다. 자기 자신 인용은 제외."""
    citations: List[Dict[str, str]] = []
    seen = set()
    for target, art_no in _iter_citations(article_text, legacy):
        if target == self_title or target.endswith(self_title):
            continue
        key = (target, art_no)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"target": target, "article_no": art_no, "kind": "citation"})
    return citations


def extract_parents_from_purpose(doc: LawDocument, legacy: bool = False) -> List[str]:
    """제1~5조에서 위임 키워드 + 「」 인용을 찾아 모법(부모 법령) 후보를 반환한다."""
    parents: List[str] = []
    seen: Set[str] = set()
    for art in doc.articles[:5]:
        body = art.get("text", "") + " " + (art.get("title") or "")
        has_delegate = any(kw in body for kw in DELEGATE_KEYWORDS)
        for target, _ in _iter_citations(body, legacy):
            if target == doc.title or target in seen:
                continue
            if has_delegate or art.get("no") == "1":
                seen.add(target)
                parents.append(target)
    return parents


def extract_delegation_relations(documents: List[LawDocument], legacy: bool = False,
                                 resolve=lambda name, doc: name) -> List[Dict[str, str]]:
    """위임 관계 (모법 → 하위 규정). 말뭉치 안에 있는 법령끼리만 연결한다."""
    titles = {d.title: d for d in documents}
    edges, seen = [], set()
    for d in documents:
        for parent in extract_parents_from_purpose(d, legacy):
            parent = resolve(parent, d)
            if parent not in titles or parent == d.title:
                continue
            if not legacy and not rank_ok(titles[parent], d):
                continue   # v1.1 : 하위 위계 문서가 상위 문서에 위임하는 역방향 관계 제거
            key = (parent, d.title)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"delegator": parent, "delegatee": d.title, "kind": "위임"})
    return edges


ADMIN_TYPES = {"AdministrativeOrder", "AdministrativeRule", "AdministrativeNotice",
               "JointAdministrativeOrder", "SpecialRegulation"}


def infer_hierarchy(documents: List[LawDocument], legacy: bool = False,
                    resolve=lambda name, doc: name) -> List[Dict[str, str]]:
    """상하위 법령 계층 추론 (다중 부모 허용).

    1단계 : 명칭 규칙  — "X 시행령/시행규칙/시행세칙" → 부모 "X"
    2단계 : 시행세칙   — 앞 두 어절이 같은 "…규칙" 을 부모로
    3단계 : 행정규칙   — 제1조(목적) 등에 인용된 모법을 부모로
    """
    titles = {d.title: d for d in documents}
    by_title = titles
    edges, seen = [], set()

    def _add(child: str, parent: str):
        key = (child, parent)
        if parent in titles and parent != child and key not in seen:
            if not legacy and not rank_ok(by_title[parent], by_title[child]):
                return   # v1.1 : 위계가 같거나 역전된 상위법령 관계 제거
            seen.add(key)
            edges.append({"child": child, "parent": parent, "kind": "상위법령"})

    for d in documents:
        added = False
        for suffix in [" 시행세칙", " 시행규칙", " 시행령"]:
            if d.title.endswith(suffix):
                parent = d.title[: -len(suffix)]
                if parent in titles:
                    _add(d.title, parent)
                    added = True
                    break
        if not added and "시행세칙" in d.title:
            tok_d = d.title.replace("시행세칙", "").strip().split()
            for cand in titles:
                if cand == d.title:
                    continue
                tok_c = cand.split()
                if len(tok_d) >= 2 and len(tok_c) >= 2 and tok_d[:2] == tok_c[:2]:
                    if cand.endswith("규칙") or cand.endswith("시행규칙"):
                        _add(d.title, cand)
                        added = True
                        break
        if added:
            continue
        if d.doc_type in ADMIN_TYPES:
            for parent in extract_parents_from_purpose(d, legacy):
                _add(d.title, resolve(parent, d))
    return edges
