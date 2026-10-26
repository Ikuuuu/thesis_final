"""LawDocument 목록 → (head, relation, tail) 트리플.

관계 12종 (온톨로지 스키마와 일치)
  type, 소관기관 / 관할법령, 발령기관, 상위법령 / 하위법령, 위임 / 위임받음,
  소속 / 포함, 인용, 외부인용
역관계 쌍(소속↔포함, 상위법령↔하위법령, 위임↔위임받음, 소관기관↔관할법령)은 함께 생성한다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .documents import LawDocument
from .normalize import EntityNormalizer, canonical_name_v10
from .rules import (
    extract_citations_from_article, extract_delegation_relations, infer_hierarchy,
)

Triple = Tuple[str, str, str]


def _article_node(law_title: str, article_no: str) -> str:
    return f"{law_title} 제{article_no}조"


def _norm(s: str) -> str:
    return " ".join(s.split()) if isinstance(s, str) else s


def build_normalizer(documents: List[LawDocument]) -> EntityNormalizer:
    mentions = []
    for d in documents:
        for art in d.articles:
            mentions += [c["target"] for c in extract_citations_from_article(art["text"], d.title)]
    return EntityNormalizer(documents, mentions)


def extract_triples(
    documents: List[LawDocument],
    legacy: bool = False,
    include_articles: bool = True,
    include_addenda: bool = False,
    include_jurisdiction: bool = True,
    include_external_citations: bool = True,
    article_level_internal_citation: bool = True,
    normalizer: Optional[EntityNormalizer] = None,
    provenance: Optional[Dict[Triple, str]] = None,
) -> List[Triple]:
    """legacy=True 이면 졸업논문 데이터셋 v1.0 과 동일한 트리플을 생성한다."""
    if legacy:
        resolve = lambda name, doc: canonical_name_v10(name)   # noqa: E731
        resolve_parent = lambda name, doc: name                 # noqa: E731
    else:
        normalizer = normalizer or build_normalizer(documents)
        resolve = resolve_parent = normalizer.resolve

    triples: List[Triple] = []
    prov = provenance if provenance is not None else {}

    def add(h, r, t, src=""):
        triples.append((h, r, t))
        if src and (h, r, t) not in prov:
            prov[(h, r, t)] = src

    titles_in_corpus = {d.title for d in documents}
    external_laws: Set[str] = set()

    for d in documents:                                         # 1. rdf:type
        add(d.title, "type", d.doc_type, f"{d.law_id}#meta")

    if include_jurisdiction:                                    # 2. 소관·발령 기관
        for d in documents:
            for org in d.jurisdiction:
                add(d.title, "소관기관", org, f"{d.law_id}#meta")
                add(org, "관할법령", d.title, f"{d.law_id}#meta")
                add(org, "type", "Organization")
            if d.issuing_org and d.issuing_org not in d.jurisdiction:
                add(d.title, "발령기관", d.issuing_org, f"{d.law_id}#filename")
                add(d.issuing_org, "type", "Organization")

    for e in infer_hierarchy(documents, legacy, resolve_parent):   # 3. 상하위 법령
        add(e["child"], "상위법령", e["parent"], "hierarchy_rule")
        add(e["parent"], "하위법령", e["child"], "hierarchy_rule")

    for e in extract_delegation_relations(documents, legacy, resolve_parent):  # 4. 위임
        add(e["delegator"], "위임", e["delegatee"], "delegation_rule")
        add(e["delegatee"], "위임받음", e["delegator"], "delegation_rule")

    if include_articles:                                        # 5. 조문·인용
        for d in documents:
            for art in d.articles:
                node = _article_node(d.title, art["no"])
                add(node, "type", "Article")
                add(node, "소속", d.title, f"{d.law_id}#제{art['no']}조")
                add(d.title, "포함", node, f"{d.law_id}#제{art['no']}조")
                for c in extract_citations_from_article(art["text"], d.title, legacy):
                    target = resolve(c["target"], d)
                    if target == d.title:
                        continue
                    src = f"{d.law_id}#제{art['no']}조"
                    if c["article_no"] and article_level_internal_citation and target in titles_in_corpus:
                        add(node, "인용", _article_node(target, c["article_no"]), src)
                    elif target in titles_in_corpus:
                        add(node, "인용", target, src)
                    elif include_external_citations:
                        add(node, "외부인용", target, src)
                        external_laws.add(target)

    if include_external_citations:                              # 6. 외부 법령 노드 타입
        for ext in sorted(external_laws):   # 정렬 → 실행마다 동일한 순서 보장
            add(ext, "type", "ExternalLegalDocument")

    if include_addenda:                                         # 7. 부칙 (선택)
        for d in documents:
            for art in d.addenda:
                node = f"{d.title} 부칙 제{art['no']}조"
                add(node, "type", "AddendumArticle")
                add(node, "소속부칙", d.title)

    triples = [(_norm(h), _norm(r), _norm(t)) for (h, r, t) in triples]
    triples = [(h, r, t) for (h, r, t) in triples if h and r and t]
    return list(dict.fromkeys(triples))
