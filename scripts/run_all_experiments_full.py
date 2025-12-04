"""按Outline原计划的trial数量跑完所有实验（耗时较长）。"""

import time

from src.experiments.group_a_baseline import run_baseline
from src.experiments.group_b_tfidf_lr import run_experiment as run_group_b
from src.experiments.group_c_embedding_lgb import run_experiment as run_group_c
from src.experiments.group_d_feature_comparison import run_experiment as run_group_d
from src.experiments.group_e_ablation import run_experiment as run_group_e


def main():
    tasks = [
        ("A", "baseline", lambda: run_baseline()),
        ("B", "grid (n=32)", lambda: run_group_b(method="grid", n_trials=32)),
        ("B", "random (n=30)", lambda: run_group_b(method="random", n_trials=30)),
        ("B", "optuna (n=40)", lambda: run_group_b(method="optuna", n_trials=40)),
        ("C", "grid (n=54)", lambda: run_group_c(method="grid", n_trials=54)),
        ("C", "random (n=30)", lambda: run_group_c(method="random", n_trials=30)),
        ("C", "optuna (n=60)", lambda: run_group_c(method="optuna", n_trials=60)),
        ("D", "tfidf_word_lr (n=50)", lambda: run_group_d(feature_model="tfidf_word_lr", n_trials=50)),
        ("D", "tfidf_word_char_lr (n=50)", lambda: run_group_d(feature_model="tfidf_word_char_lr", n_trials=50)),
        ("D", "embedding_lr (n=50)", lambda: run_group_d(feature_model="embedding_lr", n_trials=50)),
        ("D", "embedding_lgb (n=50)", lambda: run_group_d(feature_model="embedding_lgb", n_trials=50)),
        ("E", "ablation (n=30)", lambda: run_group_e(n_trials=30)),
    ]

    total_start = time.time()
    for grp, name, fn in tasks:
        start = time.time()
        print(f"[{grp}] Start {name}...")
        res = fn()
        print(f"[{grp}] Done {name} in {time.time() - start:.1f}s, summary: {res}")
    print(f"All full experiments done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
