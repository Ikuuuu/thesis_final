"""전체 파이프라인 한 번에 실행 (Windows/Linux 공통, make 불필요).

    python run_pipeline.py --quick      # 데이터 구축(v1.1) + 5 epoch 스모크 실험 + 리포트 + 추론 (약 3분)
    python run_pipeline.py --data-only  # 1~3단계만 (PDF → KG v1.1)
    python run_pipeline.py --full       # 논문 전체 스펙 재실행 (v1.0, 83런, CPU 약 1.5시간)
    python run_pipeline.py --report     # 논문 원자료로 리포트만 재생성 (수 초)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--data-only", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        run("scripts/05_report.py", "--run", "results/thesis_final_run2")
        return
    if not a.full:
        run("scripts/01_parse_pdfs.py")
        run("scripts/02_extract_triples.py")
        run("scripts/03_build_dataset.py", "--version", "v1.1")
        if a.data_only:
            return
    if a.full:
        run("scripts/04_run_experiments.py", "--data", "data/v1.0", "--out", "results/rerun")
        run("scripts/05_report.py", "--run", "results/rerun")
        run("scripts/06_inference.py", "--checkpoint", "checkpoints/rerun/Ours_s42.npz",
            "--head", "방위사업법 시행령", "--relation", "상위법령", "--topk", "5")
    else:
        run("scripts/04_run_experiments.py", "--quick")
        run("scripts/05_report.py", "--run", "results/quick")
        run("scripts/06_inference.py", "--checkpoint", "checkpoints/quick/Ours_s42.npz",
            "--head", "방위사업법 시행령", "--relation", "상위법령", "--topk", "5")


if __name__ == "__main__":
    main()
