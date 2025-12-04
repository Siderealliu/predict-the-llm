"""批量执行Outline中列出的实验（快速版，trial数量较小用于冒烟）。"""

import time

from src.experiments.group_a_baseline import run_baseline
from src.experiments.group_b_tfidf_lr import run_experiment as run_group_b
from src.experiments.group_c_embedding_lgb import run_experiment as run_group_c
from src.experiments.group_d_feature_comparison import run_experiment as run_group_d
from src.experiments.group_e_ablation import run_experiment as run_group_e


def main():
    tasks = [
        ("A", "baseline", lambda: run_baseline()),
        ("B", "grid (n=8)", lambda: run_group_b(method="grid", n_trials=8)),
        ("B", "random (n=10)", lambda: run_group_b(method="random", n_trials=10)),
        ("B", "optuna (n=15)", lambda: run_group_b(method="optuna", n_trials=15)),
        ("C", "grid (n=6)", lambda: run_group_c(method="grid", n_trials=6)),
        ("C", "random (n=10)", lambda: run_group_c(method="random", n_trials=10)),
        ("C", "optuna (n=15)", lambda: run_group_c(method="optuna", n_trials=15)),
        ("D", "tfidf_word_lr (n=15)", lambda: run_group_d(feature_model="tfidf_word_lr", n_trials=15)),
        ("D", "tfidf_word_char_lr (n=15)", lambda: run_group_d(feature_model="tfidf_word_char_lr", n_trials=15)),
        ("D", "embedding_lr (n=15)", lambda: run_group_d(feature_model="embedding_lr", n_trials=15)),
        ("D", "embedding_lgb (n=15)", lambda: run_group_d(feature_model="embedding_lgb", n_trials=15)),
        ("E", "ablation (n=10)", lambda: run_group_e(n_trials=10)),
    ]

    total_start = time.time()
    for grp, name, fn in tasks:
        start = time.time()
        print(f"[{grp}] Start {name}...")
        res = fn()
        print(f"[{grp}] Done {name} in {time.time() - start:.1f}s, result summary: {res}")
    print(f"All quick experiments done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
