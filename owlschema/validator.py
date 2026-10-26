"""스키마 기반 KG 정합성 검증.

검사 항목
  C1 타입 커버리지      : rdf:type 이 있는 엔티티 비율
  C2 domain / range     : 관계 양 끝 엔티티의 타입이 스키마 domain/range 의 하위 클래스인가
  C3 Functional         : 소속(Functional) — 조문 하나가 둘 이상의 법령에 속하는가
  C4 inverseOf 완결성   : (h, r, t) 가 있으면 (t, r⁻¹, h) 도 있는가
  C5 Transitive 폐포    : 상위법령 연쇄에서 함의되지만 명시되지 않은 트리플 수 (데이터 증강 후보)
  C6 비대칭성           : 방향성 관계에서 (h, r, t) 와 (t, r, h) 가 동시에 있는가
  C7 자기 루프          : h == t
  C8 미해결 조문 참조    : 인용 대상 조문 노드가 KG 에 존재하지 않음 (타입 없는 조문 노드)
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

from .schema import Schema

Triple = Tuple[str, str, str]
ARTICLE_RE = re.compile(r"제\d+(?:의\d+)?조$")


def validate(schema: Schema, triples: List[Triple]) -> Dict:
    T = schema.type_relation
    etype: Dict[str, str] = {}
    for h, r, t in triples:
        if r == T:
            # 여러 타입이면 더 구체적인(깊은) 클래스를 대표 타입으로
            if h not in etype or schema.depth(t) > schema.depth(etype[h]):
                etype[h] = t
    facts = [(h, r, t) for h, r, t in triples if r != T]
    fact_set = set(facts)
    ents = {h for h, _, _ in triples} | {t for _, r, t in triples if r != T}
    v = defaultdict(list)

    # C2 domain / range
    unknown = 0
    for h, r, t in facts:
        p = schema.properties.get(r)
        if p is None:
            v["C2_unknown_relation"].append((h, r, t))
            continue
        th, tt = etype.get(h), etype.get(t)
        if th is None or tt is None:
            unknown += 1
            continue
        if not schema.is_subclass(th, p["domain"]):
            v["C2_domain"].append((h, r, t, th, p["domain"]))
        if not schema.is_subclass(tt, p["range"]):
            v["C2_range"].append((h, r, t, tt, p["range"]))

    # C3 Functional
    for r, p in schema.properties.items():
        if "Functional" in p.get("characteristics", []):
            objs = defaultdict(set)
            for h, rr, t in facts:
                if rr == r:
                    objs[h].add(t)
            v["C3_functional"] += [(h, r, sorted(ts)) for h, ts in objs.items() if len(ts) > 1]

    # C4 inverse completeness
    for h, r, t in facts:
        inv = schema.inverse(r)
        if inv and (t, inv, h) not in fact_set:
            v["C4_inverse_missing"].append((h, r, t, inv))

    # C5 transitive closure (명시되지 않은 함의 트리플)
    implied = []
    for r, p in schema.properties.items():
        if "Transitive" not in p.get("characteristics", []):
            continue
        adj = defaultdict(set)
        for h, rr, t in facts:
            if rr == r:
                adj[h].add(t)
        for s in list(adj):
            seen, stack = set(), list(adj[s])
            while stack:
                x = stack.pop()
                if x in seen or x == s:
                    continue
                seen.add(x)
                stack += list(adj.get(x, ()))
            implied += [(s, r, x) for x in seen if (s, r, x) not in fact_set]
    v["C5_transitive_implied"] = implied

    # C6 asymmetry, C7 self-loop
    for h, r, t in facts:
        if h == t:
            v["C7_self_loop"].append((h, r, t))
        elif r in schema.asymmetric_hint and (t, r, h) in fact_set and h < t:
            v["C6_symmetric_pair"].append((h, r, t))

    # C8 dangling article references
    for h, r, t in facts:
        if r == "인용" and ARTICLE_RE.search(t) and t not in etype:
            v["C8_dangling_article"].append((h, r, t))

    typed = sum(1 for e in ents if e in etype)
    summary = {
        "triples": len(triples), "facts": len(facts), "entities": len(ents),
        "C1_type_coverage": typed / max(len(ents), 1),
        "C2_untyped_endpoint_facts": unknown,
        **{k: len(val) for k, val in v.items()},
    }
    return {"summary": summary, "violations": dict(v), "entity_type": etype}
