# Predict-the-LLM HPO Project

This repository contains the code and experiments for our course project on hyperparameter optimization (HPO) for the H2O.ai “Predict-the-LLM” Kaggle challenge.  
The goal is to compare different HPO strategies (Grid Search, Random Search, Optuna TPE) and feature/model combinations (TF–IDF vs embeddings, logistic regression vs LightGBM) for LLM authorship identification.

For a detailed description of the methodology and experimental results, see `main.tex`.  
For a more fine-grained implementation plan, see `Outline.md`.

---

## Dataset

The project uses the official competition dataset:

- Kaggle competition: <https://www.kaggle.com/competitions/h2oai-predict-the-llm>
- Files:
  - `train.csv`: training data with `Question`, `Response`, and `target` (0–6).
  - `test.csv`: test data with `id`, `Question`, `Response`.
  - `sample_submission.csv`: required submission format.

Place the dataset under:

```text
data/h2oai-predict-the-llm/
    train.csv
    test.csv
    sample_submission.csv
```

---

## Environment Setup

### Option 1: Conda environment (recommended)

Conda makes it easier to manage dependencies, especially GPU-enabled LightGBM.

```bash
# 1. Create conda environment (Python 3.12)
conda create -n predict-llm python=3.12 -y
conda activate predict-llm

# 2. Install core dependencies via pip
pip install numpy pandas scikit-learn matplotlib seaborn tqdm joblib

# 3. Install LightGBM with GPU support (if available)
pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON

# 4. Install Optuna and sentence-transformers
pip install optuna sentence-transformers
```

### HuggingFace mirror (optional)

If you use HuggingFace models (e.g., for sentence-transformer embeddings) in a restricted network environment, configure a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Linux/macOS
# or
set HF_ENDPOINT=https://hf-mirror.com      # Windows (cmd)
```

### GPU acceleration (optional)

If you have a GPU and want to speed up embeddings / LightGBM, set:

```bash
export USE_GPU=1
```

The code will check this environment variable where relevant.

---

## How to Run Experiments

From the project root:

- **Quick smoke run (reduced trials)**  

  ```bash
  python scripts/run_all_experiments.py
  ```

- **Full experiments (original trial budgets, slow)**  

  ```bash
  python scripts/run_all_experiments_full.py
  ```

- **Single-group examples**
  - Group B, Optuna TPE:

    ```bash
    python scripts/run_single_experiment.py --group B --method optuna --n_trials 5
    ```

  - Group C, Grid Search:

    ```bash
    python scripts/run_single_experiment.py --group C --method grid
    ```

  - Group D, Embedding + LightGBM:

    ```bash
    python scripts/run_single_experiment.py --group D --feature_model embedding_lgb --n_trials 10
    ```

- **When using HuggingFace models**

  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/run_all_experiments.py
  ```

All experiment results are written under `results/experiments/` as JSON files, one per run (e.g., `group_b_grid_*.json`, `group_d_tfidf_word_char_lr_*.json`).

---

## Project Layout

- `main.py` – optional entry point (e.g., for quick tests).
- `main.tex` – ICML-style project report.
- `Outline.md` – detailed experimental design and implementation plan.
- `scripts/`
  - `run_all_experiments.py` – quick, reduced-trial batch run.
  - `run_all_experiments_full.py` – full trial budgets for all groups (A–E).
  - `run_single_experiment.py` – run a specific group/method configuration.
- `src/config/`
  - `data_config.py` – data paths and splitting strategies (GroupKFold).
  - `model_config.py` – hyperparameter search space definitions.
  - `experiment_config.py` – configuration for experiment groups A–E.
- `src/data/`
  - `data_loader.py` – CSV loading and basic cleaning.
  - `splitter.py` – group-aware train/validation splits.
  - `preprocessor.py` – text preprocessing and QA concatenation.
- `src/features/`
  - `tfidf_features.py` – word/char TF–IDF feature builders.
  - `embedding_features.py` – sentence-transformer embeddings.
  - `statistical_features.py` – simple style/statistical features.
- `src/models/`
  - `logistic_regression.py` – multinomial logistic regression wrapper.
  - `lightgbm_model.py` – LightGBM multiclass wrapper.
- `src/hpo/`
  - `grid_search.py` – sklearn-based grid search.
  - `random_search.py` – sklearn-based random search.
  - `optuna_tpe.py` – Optuna TPE interface and pruning.
- `src/experiments/`
  - `group_a_baseline.py` – baseline TF–IDF + LR experiment.
  - `group_b_tfidf_lr.py` – TF–IDF + LR with three HPO methods.
  - `group_c_embedding_lgb.py` – embeddings + LightGBM with three HPO methods.
  - `group_d_feature_comparison.py` – feature/model combination comparison.
  - `group_e_ablation.py` – ablation study (Q vs A, stats, etc.).
- `results/`
  - `experiments/` – JSON results for each run.
  - `cache/` – cached embeddings and intermediate artifacts.
  - (optionally) `final/` – best model, submission file, summary.

---
