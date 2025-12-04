"""批量执行Outline中列出的实验。默认trial数较小，便于快速跑通。"""

from src.experiments.group_a_baseline import run_baseline
from src.experiments.group_b_tfidf_lr import run_experiment as run_group_b
from src.experiments.group_c_embedding_lgb import run_experiment as run_group_c
from src.experiments.group_d_feature_comparison import run_experiment as run_group_d
from src.experiments.group_e_ablation import run_experiment as run_group_e


def main():
    print("Running Group A baseline...")
    run_baseline()

    print("Running Group B methods (reduced trials)...")
    for method, trials in [("grid", 8), ("random", 10), ("optuna", 15)]:
        run_group_b(method=method, n_trials=trials)

    print("Running Group C methods (reduced trials)...")
    for method, trials in [("grid", 6), ("random", 10), ("optuna", 15)]:
        run_group_c(method=method, n_trials=trials)

    print("Running Group D feature comparisons (reduced trials)...")
    for fm in ["tfidf_word_lr", "tfidf_word_char_lr", "embedding_lr", "embedding_lgb"]:
        run_group_d(feature_model=fm, n_trials=15)

    print("Running Group E ablations (reduced trials)...")
    run_group_e(n_trials=10)


if __name__ == "__main__":
    main()
