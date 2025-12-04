# Predict-the-LLM 实验项目

## 数据
Kaggle竞赛数据下载：<https://www.kaggle.com/competitions/h2oai-predict-the-llm/data?select=sample_submission.csv>

## 环境安装与启动
建议使用 Python 3.12，并在项目根目录创建/使用虚拟环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需使用 HuggingFace 模型（如 sentence-transformers embedding），在网络受限环境下统一设置镜像：

```bash
export HF_ENDPOINT=http://hf-mirror.com
```

如有 GPU 并希望提速 embedding/LightGBM，在运行前设置：

```bash
export USE_GPU=1
```
（需要已安装 GPU 版 torch/LightGBM 且有可用 GPU；否则自动回退 CPU）

## 快速运行

```bash
  source .venv/bin/activate
  export HF_ENDPOINT=https://hf-mirror.com
  PYTHONPATH=. python scripts/run_all_experiments.py          # 冒烟版
  PYTHONPATH=. python scripts/run_all_experiments_full.py     # 全量版
```

- 快速冒烟版：`python scripts/run_all_experiments.py`
- 正式全量版：`python scripts/run_all_experiments_full.py`
- 若需 HF 模型，`先 export HF_ENDPOINT=https://hf-mirror.com`
- 单个实验示例：
  - 组B Optuna：`python scripts/run_single_experiment.py --group B --method optuna --n_trials 5`
  - 组C Grid：`python scripts/run_single_experiment.py --group C --method grid`
  - 组D Embedding+LGB：`python scripts/run_single_experiment.py --group D --feature_model embedding_lgb --n_trials 10`
- 批量快速跑（trial减小版）：`python scripts/run_all_experiments.py`

## 目录结构速览
- `src/config`：数据/模型/实验配置
- `src/data`：加载、预处理、划分
- `src/features`：TF-IDF、Embedding、统计特征
- `src/models`：LR、LightGBM 封装
- `src/pipeline`：特征+模型流水线
- `src/hpo`：Grid/Random/Optuna TPE
- `src/experiments`：A–E 五组实验脚本
- `scripts`：单/批量运行入口
