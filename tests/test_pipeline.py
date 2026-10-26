"""파이프라인 회귀 테스트.  python -m pytest -q   (약 30초)"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extractor.documents import LawDocument  # noqa: E402
from extractor.triples import extract_triples  # noqa: E402
from kgdata.io import load_split  # noqa: E402
from kgekit.data import load_kg  # noqa: E402
from kgekit.evaluate import evaluate  # noqa: E402
from kgekit.train import config_for, train_model  # noqa: E402
from lawparser.pdf_parser import parse_directory  # noqa: E402


@pytest.mark.skipif(not list((ROOT / "data" / "pdf").glob("*.pdf")),
                    reason="data/pdf 에 원본 PDF 가 없음 (data/pdf/README.md 참고)")
def test_v1_0_dataset_reproducible_from_pdf():
    """PDF → (v1.0 파서·규칙) → 트리플 집합이 고정된 논문 데이터셋(data/v1.0)과 같다."""
    docs = parse_directory(ROOT / "data" / "pdf", legacy=True)
    docs = [LawDocument(**{k: v for k, v in d.__dict__.items() if k != "raw_text"}) for d in docs]
    triples = set(extract_triples(docs, legacy=True))
    tr, va, te = load_split(ROOT / "data" / "v1.0")
    assert triples == set(tr) | set(va) | set(te)
    assert len(triples) == 7354


def test_training_is_deterministic():
    kg = load_kg(ROOT / "data" / "v1.0")
    cfg = config_for("Ours", seed=7, epochs=2)
    m1, P1, _ = train_model(kg, kg.train, cfg)
    m2, P2, _ = train_model(kg, kg.train, cfg)
    assert np.array_equal(P1["E"], P2["E"])
    r = evaluate(m1, P1, kg, kg.test[:50])["overall"]
    assert 0.0 < r["mrr"] <= 1.0
