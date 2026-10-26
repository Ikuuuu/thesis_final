"""단계 4 : 전체 실험 — 메인 비교 · Ablation · 데이터 축소 강건성 (중단 후 재실행 시 이어서 진행).

    python scripts/04_run_experiments.py                                   # 논문 전체 스펙 (83런, CPU 약 1.5시간)
    python scripts/04_run_experiments.py --stages main --models TransE Ours --seeds 42
    python scripts/04_run_experiments.py --quick                           # 5 epoch · seed 42 스모크 (약 2분)
    python scripts/04_run_experiments.py --data data/v1.1 --out results/v1_1    # 개선 데이터셋으로

Stage main       : TransE, RotatE, DistMult, ComplEx, TKRL, TransC, TransO, Ours, OursC × 5 seed
Stage ablation   : Ours −L_dir / −L_hier_attn / −L_attr / −L_type × 5 seed
Stage robustness : TransE, Ours × train {25, 50, 75}% × 3 seed
결과 : <out>/raw/<id>_s<seed>.json + .npz, seed 42 모델 파라미터는 checkpoints/<out이름>/ 에 저장
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgekit.data import load_kg, subsample_train  # noqa: E402
from kgekit.evaluate import evaluate  # noqa: E402
from kgekit.train import config_for, train_chunk  # noqa: E402

MAIN = ["TransE", "RotatE", "DistMult", "ComplEx", "TKRL", "TransC", "TransO", "Ours", "OursC"]
ABL = {"Ours-L_dir": {"delta_dir": 0.0}, "Ours-L_hier_attn": {"beta_hier": 0.0},
       "Ours-L_attr": {"gamma_attr": 0.0}, "Ours-L_type": {"alpha_type": 0.0}}


def build_jobs(stages, models, seeds, rob_seeds, fracs):
    jobs = []
    if "main" in stages:
        jobs += [dict(stage="main", id=f"main_{m}", model=m, variant=m, seed=s, fraction=1.0, ov={})
                 for m in models for s in seeds]
    if "ablation" in stages:
        jobs += [dict(stage="ablation", id=f"abl_{n}", model="Ours", variant=n, seed=s, fraction=1.0, ov=ov)
                 for n, ov in ABL.items() for s in seeds]
    if "robustness" in stages:
        jobs += [dict(stage="robustness", id=f"rob_{m}_f{int(f * 100)}", model=m, variant=m, seed=s,
                      fraction=f, ov={}) for m in ["TransE", "Ours"] for f in fracs for s in rob_seeds]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1.0")
    ap.add_argument("--out", default="results/rerun")
    ap.add_argument("--stages", nargs="+", default=["main", "ablation", "robustness"])
    ap.add_argument("--models", nargs="+", default=MAIN)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--rob-seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="epochs=5, seed 42, 모델 4종만")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.seeds, args.rob_seeds = 5, [42], [42]
        args.models = ["TransE", "TransO", "Ours", "OursC"]
        args.out = args.out if args.out != "results/rerun" else "results/quick"

    kg = load_kg(ROOT / args.data)
    out = ROOT / args.out
    raw = out / "raw"
    ck = ROOT / "checkpoints" / out.name
    raw.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.stages, args.models, args.seeds, args.rob_seeds, args.fracs)
    (out / "manifest.json").write_text(json.dumps({"data": args.data, "epochs": args.epochs, "n_jobs": len(jobs),
                                                   "seeds": args.seeds, "stages": args.stages}, ensure_ascii=False,
                                                  indent=1), encoding="utf-8")
    print(f"[4] {args.data}: |E|={kg.num_ent} train={len(kg.train)} test={len(kg.test)} — 작업 {len(jobs)}개")
    for k, j in enumerate(jobs, 1):
        f = raw / f"{j['id']}_s{j['seed']}.json"
        if f.exists():
            continue
        t0 = time.time()
        train = kg.train if j["fraction"] >= 1.0 else subsample_train(kg.train, j["fraction"], seed=j["seed"])
        cfg = config_for(j["model"], seed=j["seed"], epochs=args.epochs, **j["ov"])
        model, P, _, _ = train_chunk(kg, train, cfg)
        res = evaluate(model, P, kg, kg.test)
        np.savez_compressed(f.with_suffix(".npz"), rr=res["_rr"], rel_of=res["_rel_of"],
                            dir_mask=res["_dir_mask"], hier_mask=res["_hier_mask"])
        rec = {k2: v for k2, v in j.items() if k2 != "ov"} | {"overrides": j["ov"]}
        rec |= {k2: v for k2, v in res.items() if not k2.startswith("_")}
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        if j["stage"] == "main" and j["seed"] == 42:
            np.savez_compressed(ck / f"{j['model']}_s42.npz", **{k2: v for k2, v in P.items()})
        print(f"  [{k}/{len(jobs)}] {j['id']:22s} s{j['seed']}  MRR={res['overall']['mrr']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"    → {args.out}/raw  (집계: python scripts/05_report.py --run {args.out})")


if __name__ == "__main__":
    main()
