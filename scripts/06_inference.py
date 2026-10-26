"""단계 6 : 추론 데모 — 링크 예측 질의와 유사 엔티티 검색.

    # (h, r, ?) : 「방위사업법 시행령」의 상위법령은?
    python scripts/06_inference.py --head "방위사업법 시행령" --relation 상위법령
    # (?, r, t) : 「방위사업법」을 상위법령으로 두는 문서는?
    python scripts/06_inference.py --tail "방위사업법" --relation 상위법령
    # 임베딩 공간에서 가까운 엔티티
    python scripts/06_inference.py --similar "군수품조달관리규정"

체크포인트는 04_run_experiments.py 가 저장한 checkpoints/<run>/<model>_s42.npz 를 쓴다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgekit.data import load_kg  # noqa: E402
from kgekit.models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1.0")
    ap.add_argument("--checkpoint", default="checkpoints/demo_seed42/Ours_s42.npz")
    ap.add_argument("--model", default=None, help="기본: 체크포인트 파일명에서 추론")
    ap.add_argument("--head")
    ap.add_argument("--tail")
    ap.add_argument("--relation")
    ap.add_argument("--similar")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--hide-known", action="store_true", help="이미 KG 에 있는 정답은 제외(새 관계 후보만)")
    args = ap.parse_args()

    kg = load_kg(ROOT / args.data)
    ck = ROOT / args.checkpoint
    if not ck.exists():
        sys.exit(f"체크포인트 없음: {args.checkpoint}\n먼저 python scripts/04_run_experiments.py 를 실행하세요 "
                 f"(빠르게: --quick → checkpoints/quick/).")
    P = dict(np.load(ck))
    name = args.model or ck.stem.split("_s")[0]
    model = build_model(name)
    if hasattr(model, "set_context"):
        model.set_context(kg)
    if hasattr(model, "prepare_eval"):
        model.prepare_eval(P)
    id2e = {v: k for k, v in kg.ent2id.items()}
    ne = kg.num_ent

    def eid(s):
        if s not in kg.ent2id:
            cands = [e for e in kg.ent2id if s in e][:5]
            sys.exit(f"엔티티 '{s}' 없음. 비슷한 이름: {cands}")
        return kg.ent2id[s]

    if args.relation:
        r = kg.rel2id[args.relation]
        if args.head:
            h = eid(args.head)
            e = model.energy(P, np.full(ne, h), np.full(ne, r), np.arange(ne))
            known = {t for (hh, rr, t) in kg.all_triples if hh == h and rr == r}
            title = f"({args.head}, {args.relation}, ?)"
            self_id = h
        else:
            t = eid(args.tail)
            e = model.energy(P, np.arange(ne), np.full(ne, r), np.full(ne, t))
            known = {hh for (hh, rr, tt) in kg.all_triples if tt == t and rr == r}
            title = f"(?, {args.relation}, {args.tail})"
            self_id = t
        print(f"[{name}] {title}  — 에너지 낮을수록 그럴듯함, ★ = KG 에 이미 있는 관계")
        shown = 0
        for i in np.argsort(e):
            if i == self_id or (args.hide_known and i in known):
                continue   # 질의 엔티티 자신은 제외
            print(f"  {shown + 1:2d}. {'★' if i in known else ' '} {id2e[int(i)]}  ({e[i]:.3f})")
            shown += 1
            if shown >= args.topk:
                break
    if args.similar:
        E = P["E"] / np.maximum(np.linalg.norm(P["E"], axis=1, keepdims=True), 1e-12)
        q = eid(args.similar)
        sims = E @ E[q]
        print(f"[{name}] '{args.similar}' 와 코사인 유사도가 높은 엔티티")
        for k, i in enumerate([i for i in np.argsort(-sims) if i != q][: args.topk], 1):
            print(f"  {k:2d}. {id2e[int(i)]}  ({sims[i]:.3f})")


if __name__ == "__main__":
    main()
