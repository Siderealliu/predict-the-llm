# Predict-the-LLM 实验项目

## 数据
Kaggle竞赛数据下载：<https://www.kaggle.com/competitions/h2oai-predict-the-llm/data?select=sample_submission.csv>

## 环境安装与启动

### 方案一：Conda 环境（推荐 ⭐）

Conda可以更简单地管理复杂依赖，特别是GPU版本的LightGBM。

```bash
# 1. 创建conda环境（Python 3.12）
conda create -n predict-llm python=3.12 -y
conda activate predict-llm

# 2. 安装主要依赖（使用pip，更快更稳定）
pip install numpy pandas scikit-learn matplotlib seaborn tqdm joblib

# 3. 安装LightGBM（使用conda，避免GPU版OpenCL问题）
# CPU版本
conda install -c conda-forge lightgbm -y
# GPU版本（需要NVIDIA GPU + CUDA）
conda install -c conda-forge lightgbm-gpu -y

# 4. 安装Optuna和sentence-transformers
pip install optuna sentence-transformers

# 5. 安装项目（开发模式）
pip install -e .
```

**Conda方案优点**：
- ✅ 仅用conda安装LightGBM，避免GPU版OpenCL依赖问题
- ✅ 其他依赖使用pip，安装更快更稳定
- ✅ 依赖隔离好，减少冲突
- ✅ 支持多版本Python切换
- ✅ 跨平台（Windows/macOS/Linux）一致性更好

### 方案二：Python venv 环境

```bash
# 1. 创建venv
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows

# 2. 安装依赖
pip install -e .

# 3. 如需GPU版LightGBM（更复杂）
pip uninstall lightgbm -y
# Ubuntu/Debian需先安装OpenCL：
# sudo apt install intel-opencl-icd
# 然后安装GPU版：
# pip install lightgbm --install-option=--precompile
```

**Venv方案优点**：
- ✅ Python原生，无需额外工具
- ✅ 轻量级，下载快
- ✅ 适合熟悉pip生态的用户

### HuggingFace 镜像配置

如需使用 HuggingFace 模型（如 sentence-transformers embedding），在网络受限环境下统一设置镜像：

```bash
# 方案1：临时设置（推荐）
export HF_ENDPOINT=https://hf-mirror.com  # Linux/macOS
# 或
set HF_ENDPOINT=https://hf-mirror.com      # Windows

# 方案2：永久配置
echo "https://hf-mirror.com" > ~/.cache/huggingface/transformers/../mirror.txt
```

### GPU 加速配置

如有 GPU 并希望提速 embedding/LightGBM，在运行前设置：

```bash
export USE_GPU=1
```

**GPU使用说明**：
- **CPU模式**：所有实验默认使用CPU，已安装GPU库时自动检测
- **GPU模式**：设置`USE_GPU=1`环境变量启用
- **自动回退**：如果GPU不可用或配置错误，自动回退到CPU模式
- **LightGBM GPU要求**：需要NVIDIA GPU + CUDA，或安装`lightgbm-gpu`（conda）

**常见GPU问题解决**：
```bash
# 检查GPU是否可用
nvidia-smi

# 如果LightGBM报错"没有opencl设备"：
# 方案A：使用conda安装GPU版（推荐）
conda install -c conda-forge lightgbm-gpu -y

# 方案B：安装OpenCL运行时（Linux）
sudo apt install intel-opencl-icd  # Ubuntu/Debian

# 方案C：强制使用CPU（保险）
export USE_GPU=0  # 或在运行脚本时添加 --force-cpu 参数）
```

### 环境方案对比

| 特性 | Conda ⭐ (混合模式) | venv |
|------|---------------------|------|
| **GPU支持** | ✅ 简单，仅conda装LightGBM | ⚠️ 需手动安装OpenCL |
| **安装速度** | ✅ 快（主要依赖用pip） | ✅ 快速 |
| **依赖管理** | ✅ LightGBM隔离好 | ⚠️ 可能冲突 |
| **跨平台性** | ✅ 一致性好 | ✅ 好 |
| **学习成本** | ⚠️ 需了解conda命令 | ✅ 简单 |
| **推荐场景** | 需要GPU或科研环境 | 开发、测试、轻量级 |

**选择建议**：
- 🥇 **需要GPU加速**：推荐Conda混合模式
- 🥈 **熟悉Python生态**：可选venv
- 🥉 **纯CPU使用**：venv更简单快捷

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
