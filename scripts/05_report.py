"""단계 5 : 결과 집계 — 비교표 · 가설 판정(H1/H2/H3) · 그림 · 요약 리포트.

    python scripts/05_report.py --run results/thesis_final_run2     # 논문 본 실험 원자료
    python scripts/05_report.py --run results/rerun                 # 재실행 결과

산출물 (<run>/summary/) : main_results.csv, ablation.csv, robustness.csv, per_relation.csv,
                         hypotheses.json, report.md, main_comparison.png, ablation.png, robustness.png
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgekit.stats import paired_bootstrap  # noqa: E402

ORDER = ["TransE", "RotatE", "DistMult", "ComplEx", "TKRL", "TransC", "TransO", "Ours", "OursC"]
LABEL = {"OursC": "Ours(TransC base)"}


def load(run):
    R = defaultdict(list)
    for p in sorted((run / "raw").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        z = np.load(p.with_suffix(".npz"))
        r["_rr"], r["_dir"], r["_hier"] = z["rr"], z["dir_mask"], z["hier_mask"]
        R[r["id"]].append(r)
    return R


def agg(rs, *path):
    vals = []
    for r in rs:
        x = r
        for k in path:
            x = x[k]
        vals.append(x)
    return statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/thesis_final_run2")
    args = ap.parse_args()
    run = ROOT / args.run
    R = load(run)
    out = run / "summary"
    out.mkdir(exist_ok=True)
    rr = lambda i: np.mean([r["_rr"] for r in R[i]], axis=0)  # noqa: E731

    main_rows = []
    for m in ORDER:
        rs = R.get(f"main_{m}")
        if not rs:
            continue
        mrr, sd = agg(rs, "overall", "mrr")
        main_rows.append({"model": LABEL.get(m, m), "MRR": round(mrr, 4), "MRR_std": round(sd, 4),
                          **{k.replace("hits", "Hits"): round(agg(rs, "overall", k)[0], 4)
                             for k in ["hits@1", "hits@3", "hits@10"]},
                          "dir_MRR": round(agg(rs, "groups", "directional", "mrr")[0], 4),
                          "dir_Hits@1": round(agg(rs, "groups", "directional", "hits@1")[0], 4),
                          "hier_MRR": round(agg(rs, "groups", "hierarchy", "mrr")[0], 4), "n_seeds": len(rs)})
    write_csv(out / "main_results.csv", main_rows)

    per_rel = []
    for m in ORDER:
        rs = R.get(f"main_{m}")
        for rel in (rs[0]["per_relation"] if rs else {}):
            per_rel.append({"model": LABEL.get(m, m), "relation": rel,
                            "MRR": round(statistics.mean(r["per_relation"][rel]["mrr"] for r in rs), 4),
                            "Hits@1": round(statistics.mean(r["per_relation"][rel]["hits@1"] for r in rs), 4),
                            "n": rs[0]["per_relation"][rel]["n"]})
    write_csv(out / "per_relation.csv", per_rel)

    abl_rows = []
    for v in ["main_Ours", "abl_Ours-L_dir", "abl_Ours-L_hier_attn", "abl_Ours-L_attr", "abl_Ours-L_type"]:
        if R.get(v):
            abl_rows.append({"variant": "Ours(full)" if v == "main_Ours" else v[4:],
                             "MRR": round(agg(R[v], "overall", "mrr")[0], 4),
                             "dir_MRR": round(agg(R[v], "groups", "directional", "mrr")[0], 4),
                             "hier_MRR": round(agg(R[v], "groups", "hierarchy", "mrr")[0], 4)})
    write_csv(out / "ablation.csv", abl_rows)

    rob_rows, rob = [], defaultdict(dict)
    for m in ["TransE", "Ours"]:
        if R.get(f"main_{m}"):
            rob[m][1.0] = agg(R[f"main_{m}"], "overall", "mrr")[0]
        for i, rs in R.items():
            if i.startswith(f"rob_{m}_f"):
                rob[m][rs[0]["fraction"]] = agg(rs, "overall", "mrr")[0]
        for f in sorted(rob[m]):
            rob_rows.append({"model": m, "train_fraction": f, "MRR": round(rob[m][f], 4)})
    write_csv(out / "robustness.csv", rob_rows)

    hyp = {}
    if R.get("main_Ours") and R.get("main_TransO"):
        dm = R["main_Ours"][0]["_dir"]
        bs = paired_bootstrap(rr("main_Ours")[dm], rr("main_TransO")[dm])
        d = (agg(R["main_Ours"], "groups", "directional", "hits@1")[0]
             - agg(R["main_TransO"], "groups", "directional", "hits@1")[0]) * 100
        hyp["H1"] = {"criterion": "방향성 Hits@1 ≥ TransO+5%p & p<0.05", "delta_hits1_pp": round(d, 2),
                     "bootstrap": bs, "passed": bool(d >= 5 and bs["p_value"] < 0.05)}
    if R.get("main_Ours") and R.get("abl_Ours-L_hier_attn"):
        d = (agg(R["main_Ours"], "groups", "hierarchy", "mrr")[0]
             - agg(R["abl_Ours-L_hier_attn"], "groups", "hierarchy", "mrr")[0]) * 100
        hyp["H2"] = {"criterion": "−L_hier_attn 시 계층 MRR ≥3%p 저하", "drop_pp": round(d, 2),
                     "n_hier_test": int(R["main_Ours"][0]["_hier"].sum()), "passed": bool(d >= 3)}
    if 0.5 in rob.get("Ours", {}) and 0.5 in rob.get("TransE", {}) and 1.0 in rob["Ours"]:
        do, dt = rob["Ours"][1.0] - rob["Ours"][0.5], rob["TransE"][1.0] - rob["TransE"][0.5]
        hyp["H3"] = {"criterion": "50% 축소 ΔMRR(Ours) < 0.7×ΔMRR(TransE)", "delta_ours": round(do, 4),
                     "threshold": round(0.7 * dt, 4), "passed": bool(do < 0.7 * dt)}
    (out / "hypotheses.json").write_text(json.dumps(hyp, ensure_ascii=False, indent=1), encoding="utf-8")

    L = [f"# 실험 결과 요약 — {args.run}", "", "## 1. 메인 비교 (seed 평균)", "",
         "| 모델 | MRR | Hits@1 | Hits@10 | 방향성 MRR | 방향성 H@1 | 계층 MRR |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['model']} | {r['MRR']:.4f} | {r['Hits@1']:.4f} | {r['Hits@10']:.4f} | {r['dir_MRR']:.4f} | "
          f"{r['dir_Hits@1']:.4f} | {r['hier_MRR']:.4f} |" for r in main_rows]
    L += ["", "## 2. 가설 판정", ""]
    for k, h in hyp.items():
        L.append(f"- **{k}** ({h['criterion']}) : {'✅ 충족' if h['passed'] else '❌ 미충족'}  "
                 f"`{json.dumps({x: y for x, y in h.items() if x not in ('criterion', 'passed', 'bootstrap')}, ensure_ascii=False)}`")
    if abl_rows:
        L += ["", "## 3. Ablation", "", "| 변형 | MRR | 방향성 MRR | 계층 MRR |", "|---|---|---|---|"]
        L += [f"| {r['variant']} | {r['MRR']:.4f} | {r['dir_MRR']:.4f} | {r['hier_MRR']:.4f} |" for r in abl_rows]
    if rob_rows:
        L += ["", "## 4. 데이터 축소 강건성 (MRR)", "", "| 비율 | TransE | Ours |", "|---|---|---|"]
        for f in sorted({r["train_fraction"] for r in rob_rows}):
            L.append(f"| {int(f * 100)}% | {rob['TransE'].get(f, float('nan')):.4f} | {rob['Ours'].get(f, float('nan')):.4f} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if main_rows:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        x = np.arange(len(main_rows))
        ax.bar(x - 0.27, [r["MRR"] for r in main_rows], 0.27, yerr=[r["MRR_std"] for r in main_rows],
               capsize=2, label="MRR")
        ax.bar(x, [r["dir_MRR"] for r in main_rows], 0.27, label="Directional MRR")
        ax.bar(x + 0.27, [r["hier_MRR"] for r in main_rows], 0.27, label="Hierarchy MRR")
        ax.set_xticks(x)
        ax.set_xticklabels([r["model"].replace("(TransC base)", "\n(TransC)") for r in main_rows], fontsize=8)
        ax.legend(fontsize=8)
        ax.set_title("Main comparison (Defense-Law KG, test)")
        fig.tight_layout()
        fig.savefig(out / "main_comparison.png", dpi=150)
        plt.close(fig)
    if abl_rows:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(len(abl_rows))
        ax.bar(x - 0.2, [r["MRR"] for r in abl_rows], 0.4, label="MRR")
        ax.bar(x + 0.2, [r["hier_MRR"] for r in abl_rows], 0.4, label="Hierarchy MRR")
        ax.set_xticks(x)
        ax.set_xticklabels([r["variant"].replace("Ours-", "−") for r in abl_rows], fontsize=8)
        ax.legend(fontsize=8)
        ax.set_title("Ablation")
        fig.tight_layout()
        fig.savefig(out / "ablation.png", dpi=150)
        plt.close(fig)
    if rob_rows:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for m in rob:
            fs = sorted(rob[m])
            ax.plot([f * 100 for f in fs], [rob[m][f] for f in fs], marker="o", label=m)
        ax.set_xlabel("training data (%)")
        ax.set_ylabel("MRR")
        ax.legend()
        ax.set_title("Robustness to data reduction")
        fig.tight_layout()
        fig.savefig(out / "robustness.png", dpi=150)
        plt.close(fig)
    print("\n".join(L))
    print(f"\n→ {args.run}/summary/")


if __name__ == "__main__":
    main()
