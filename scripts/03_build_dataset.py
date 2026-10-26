"""단계 3 : 트리플 → OWL Lite 스키마 검증 → train/valid/test 분할 (관계별 층화 80/10/10).

    python scripts/03_build_dataset.py --version v1.1
※ 논문 데이터셋 v1.0 의 분할은 data/v1.0 에 고정본으로 들어 있다 (재분할하지 않음).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgdata.io import build_id_mappings, save_split  # noqa: E402
from kgdata.splitter import stratified_split  # noqa: E402
from owlschema.schema import load_schema  # noqa: E402
from owlschema.validator import validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.1")
    ap.add_argument("--triples", default=None, help="기본: data/interim/triples_<version>.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    triples = sorted({tuple(x) for x in json.loads(
        (ROOT / (args.triples or f"data/interim/triples_{args.version}.json")).read_text(encoding="utf-8"))})
    s = validate(load_schema(ROOT / "ontology" / "schema.json"), triples)["summary"]
    print(f"[3] 스키마 검증 : 타입 커버리지 {s['C1_type_coverage'] * 100:.1f}%, "
          f"domain/range 위반 {s.get('C2_domain', 0) + s.get('C2_range', 0)}, "
          f"역관계 누락 {s.get('C4_inverse_missing', 0)}, 방향성 양방향 쌍 {s.get('C6_symmetric_pair', 0)}")
    split = stratified_split(triples, 0.8, 0.1, 0.1, seed=args.seed, stratify_by="relation")
    e2i, r2i = build_id_mappings(triples)
    out = ROOT / "data" / args.version
    if args.version == "v1.0" and (out / "train.tsv").exists():
        sys.exit("data/v1.0 은 논문 고정본입니다. 재분할하려면 "
                 "--version v1.0_rebuild --triples data/interim/triples_v1.0.json 처럼 다른 이름을 쓰세요.")
    save_split(out, split, e2i, r2i)
    print(f"    → data/{args.version}: train={len(split[0])}, valid={len(split[1])}, test={len(split[2])}, "
          f"|E|={len(e2i)}, |R|={len(r2i)}")


if __name__ == "__main__":
    main()
