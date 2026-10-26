"""단계 2 : 문서 JSON → 트리플 (규칙 기반 추출 + 엔티티 정규화 + 위계 검증).

    python scripts/02_extract_triples.py                        # v1.1
    python scripts/02_extract_triples.py --legacy               # v1.0 (논문 데이터셋과 동일한 트리플 집합)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extractor.documents import load_documents  # noqa: E402
from extractor.triples import extract_triples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--raw", default=None)
    args = ap.parse_args()
    ver = "v1.0" if args.legacy else "v1.1"
    raw = ROOT / (args.raw or ("data/raw_v1_0" if args.legacy else "data/raw"))
    triples = extract_triples(load_documents(raw), legacy=args.legacy)
    out = ROOT / "data" / "interim" / f"triples_{ver}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(triples, ensure_ascii=False, indent=0), encoding="utf-8")
    rel = Counter(r for _, r, _ in triples)
    print(f"[2] {ver} 트리플 {len(triples):,}개 → {out.relative_to(ROOT)}")
    print("    " + ", ".join(f"{k}={v}" for k, v in rel.most_common()))


if __name__ == "__main__":
    main()
