"""ID 매핑 생성, TSV 저장·로드."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

Triple = Tuple[str, str, str]

REL_EN = {
    "type": "type", "소속": "belongsTo", "포함": "contains", "외부인용": "citesExternal",
    "인용": "cites", "소관기관": "competentAuthority", "관할법령": "governingLaw",
    "위임": "delegates", "위임받음": "delegatedBy", "상위법령": "parentLaw",
    "하위법령": "childLaw", "발령기관": "issuedBy",
}
INVERSE = {"소속": "포함", "포함": "소속", "위임": "위임받음", "위임받음": "위임",
           "상위법령": "하위법령", "하위법령": "상위법령", "소관기관": "관할법령", "관할법령": "소관기관"}


def build_id_mappings(triples: List[Triple]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """빈도 내림차순으로 ID 부여 (논문 데이터셋과 동일 규칙)."""
    ent, rel = Counter(), Counter()
    for h, r, t in triples:
        ent[h] += 1
        ent[t] += 1
        rel[r] += 1
    return ({e: i for i, (e, _) in enumerate(ent.most_common())},
            {r: i for i, (r, _) in enumerate(rel.most_common())})


def save_split(out_dir: Path, split, e2i, r2i) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ts in zip(["train", "valid", "test"], split):
        with (out_dir / f"{name}.tsv").open("w", encoding="utf-8") as f:
            for h, r, t in ts:
                f.write(f"{e2i[h]}\t{r2i[r]}\t{e2i[t]}\n")
        with (out_dir / f"{name}_labeled.tsv").open("w", encoding="utf-8") as f:
            for h, r, t in ts:
                f.write(f"{h}\t{r}\t{t}\n")
    for name, m in [("entity2id", e2i), ("relation2id", r2i)]:
        with (out_dir / f"{name}.tsv").open("w", encoding="utf-8") as f:
            for k, v in m.items():
                f.write(f"{k}\t{v}\n")


def load_split(d: Path):
    """→ (train, valid, test) 문자열 트리플 리스트."""
    def rmap(p):
        m = {}
        for line in p.open(encoding="utf-8"):
            a = line.rstrip("\n").split("\t")
            if len(a) >= 2:
                m[int(a[1])] = a[0]
        return m
    i2e, i2r = rmap(d / "entity2id.tsv"), rmap(d / "relation2id.tsv")
    out = []
    for name in ["train", "valid", "test"]:
        ts = []
        for line in (d / f"{name}.tsv").open(encoding="utf-8"):
            a = line.split()
            if len(a) == 3:
                ts.append((i2e[int(a[0])], i2r[int(a[1])], i2e[int(a[2])]))
        out.append(ts)
    return tuple(out)
