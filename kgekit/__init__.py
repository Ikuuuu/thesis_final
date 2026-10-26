"""kgekit — NumPy 기반 경량 지식그래프 임베딩(KGE) 실험 엔진.

GPU·PyTorch 없이 CPU(NumPy)만으로 KGE 학습·평가·통계검정을 수행한다.
폐쇄망(air-gapped) 환경에서도 동일 결과를 재현할 수 있도록 외부 의존성을 최소화했다.

- data     : KG 로드, 관계 그룹(방향성/계층/역관계) 정의, 학습 데이터 축소 샘플링
- models   : TransE / RotatE / DistMult / ComplEx / TransO / TKRL / TransC / Ours / OursC
- train    : margin-ranking 학습 + 온톨로지 보조 손실(L_dir, L_hier_attn, L_attr, L_type)
- evaluate : Filtered MRR, Hits@1/3/10 (전체·관계별·관계그룹별)
- stats    : paired bootstrap 검정, 가설 H1/H2/H3 판정
"""
__all__ = ["data", "models", "train", "evaluate", "stats"]
__version__ = "1.0.0"
