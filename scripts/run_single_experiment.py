"""按组别运行单个实验，便于命令行调用。"""

import argparse

from src.experiments.group_a_baseline import run_baseline
from src.experiments.group_b_tfidf_lr import run_experiment as run_group_b
from src.experiments.group_c_embedding_lgb import run_experiment as run_group_c
from src.experiments.group_d_feature_comparison import run_experiment as run_group_d
from src.experiments.group_e_ablation import run_experiment as run_group_e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--method", default=None, help="For group B/C methods")
    parser.add_argument("--feature_model", default=None, help="For group D feature_model")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--force_recompute", action="store_true", help="Force recompute embeddings")
    args = parser.parse_args()

    if args.group == "A":
        res = run_baseline()
    elif args.group == "B":
        res = run_group_b(method=args.method or "grid", n_trials=args.n_trials)
    elif args.group == "C":
        res = run_group_c(method=args.method or "grid", n_trials=args.n_trials, force_recompute=args.force_recompute)
    elif args.group == "D":
        fm = args.feature_model or "tfidf_word_lr"
        res = run_group_d(feature_model=fm, n_trials=args.n_trials or 50)
    else:
        res = run_group_e(n_trials=args.n_trials or 30)
    print(res)


if __name__ == "__main__":
    main()
