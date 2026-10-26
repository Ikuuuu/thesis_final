"""스키마 → OWL(Turtle) 변환 + OWL Lite 프로파일 점검, KG 인스턴스 → RDF 변환."""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple
from urllib.parse import quote

from .schema import Schema

# OWL Lite 에서 허용되는 속성 특성 (AsymmetricProperty 등 OWL 2 구성요소는 불가)
OWL_LITE_CHARS = {"Functional", "InverseFunctional", "Transitive", "Symmetric"}
# OWL Lite 에서 금지된 클래스 구성자
OWL_LITE_FORBIDDEN = {"unionOf", "complementOf", "oneOf", "hasValue", "disjointWith"}


def check_owl_lite(schema: Schema) -> List[str]:
    """OWL Lite 프로파일 위반 목록 (빈 리스트면 통과)."""
    problems = []
    for name, p in schema.properties.items():
        for ch in p.get("characteristics", []):
            if ch not in OWL_LITE_CHARS:
                problems.append(f"{name}: '{ch}' 는 OWL Lite 속성 특성이 아님")
        for key in ("domain", "range"):
            v = p.get(key)
            if isinstance(v, list):
                problems.append(f"{name}: {key} 가 여러 클래스 → unionOf 필요 (OWL Lite 불가)")
            elif v and v not in schema.classes:
                problems.append(f"{name}: {key} 클래스 '{v}' 미정의")
        inv = p.get("inverse")
        if inv and schema.properties.get(inv, {}).get("inverse") != name:
            problems.append(f"{name}: inverseOf '{inv}' 가 상호 선언되지 않음")
    for c, meta in schema.classes.items():
        for k in meta:
            if k in OWL_LITE_FORBIDDEN:
                problems.append(f"{c}: '{k}' 는 OWL Lite 에서 사용할 수 없음")
    return problems


def schema_to_graph(schema: Schema):
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS, XSD  # noqa: F401

    g = rdflib.Graph()
    NS = rdflib.Namespace(schema.namespace)
    g.bind("dl", NS)
    g.bind("owl", OWL)
    onto = rdflib.URIRef(schema.namespace.rstrip("#"))
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, rdflib.Literal("국방 법령 지식그래프 온톨로지 (OWL Lite)", lang="ko")))

    for c, meta in schema.classes.items():
        g.add((NS[c], RDF.type, OWL.Class))
        g.add((NS[c], RDFS.label, rdflib.Literal(meta.get("label", c), lang="ko")))
        if meta.get("parent"):
            g.add((NS[c], RDFS.subClassOf, NS[meta["parent"]]))

    char_map = {"Functional": OWL.FunctionalProperty, "InverseFunctional": OWL.InverseFunctionalProperty,
                "Transitive": OWL.TransitiveProperty, "Symmetric": OWL.SymmetricProperty}
    for name, p in schema.properties.items():
        u = NS[p["iri"]]
        g.add((u, RDF.type, OWL.ObjectProperty))
        g.add((u, RDFS.label, rdflib.Literal(name, lang="ko")))
        g.add((u, RDFS.domain, NS[p["domain"]]))
        g.add((u, RDFS.range, NS[p["range"]]))
        for ch in p.get("characteristics", []):
            g.add((u, RDF.type, char_map[ch]))
        if p.get("inverse"):
            g.add((u, OWL.inverseOf, NS[schema.properties[p["inverse"]]["iri"]]))
    return g


def _ent_iri(ns: str, name: str) -> str:
    return ns + "e/" + quote(re.sub(r"\s+", "_", name), safe="")


def kg_to_graph(schema: Schema, triples: Iterable[Tuple[str, str, str]]):
    """KG 인스턴스(트리플) → RDF 그래프. 엔티티 IRI 는 명칭을 URL 인코딩해 만든다."""
    import rdflib
    from rdflib.namespace import RDF, RDFS

    g = schema_to_graph(schema)
    NS = rdflib.Namespace(schema.namespace)
    labeled = set()
    for h, r, t in triples:
        hu = rdflib.URIRef(_ent_iri(schema.namespace, h))
        if h not in labeled:
            g.add((hu, RDFS.label, rdflib.Literal(h, lang="ko")))
            labeled.add(h)
        if r == schema.type_relation:
            g.add((hu, RDF.type, NS[t]))
            continue
        tu = rdflib.URIRef(_ent_iri(schema.namespace, t))
        if t not in labeled:
            g.add((tu, RDFS.label, rdflib.Literal(t, lang="ko")))
            labeled.add(t)
        g.add((hu, NS[schema.properties[r]["iri"]] if r in schema.properties else NS[r], tu))
    return g
