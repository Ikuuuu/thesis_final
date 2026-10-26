"""단계 1 : 법령 PDF → 문서별 JSON (lawparser).

    python scripts/01_parse_pdfs.py              # v1.1 파서 → data/raw
    python scripts/01_parse_pdfs.py --legacy     # v1.0 파서(논문 데이터셋) → data/raw_v1_0
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lawparser.pdf_parser import parse_directory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/pdf")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = ROOT / (args.out or ("data/raw_v1_0" if args.legacy else "data/raw"))
    out.mkdir(parents=True, exist_ok=True)
    docs = parse_directory(ROOT / args.pdf_dir, legacy=args.legacy)
    for d in docs:
        rec = asdict(d)
        rec.pop("raw_text", None)
        (out / f"{d.law_id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[1] 문서 {len(docs)}건, 조문 {sum(len(d.articles) for d in docs):,}개 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
