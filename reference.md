# 超参数优化（HPO）实验

## Original

Delve deep into the world of Language Learning Models (LLMs)! The objective of this competition is to detect which out of 7 possible LLM models produced a particular output. With each model having its unique subtleties and quirks, can you identify which one generated the response?

The challenge of pinpointing the origin LLM of a given output is not only intriguing but is also an area of spirited research. Understanding the nuances between different models allows for advancements in model differentiation, quality assessment, and potential applications in multiple sectors. Dive in, explore the intricacies, and contribute to this emerging area of study!

Loss Function: Multi-class Log Loss

$$
[ \text{LogLoss} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \log(p_{ij}) ]
$$

Submission File Format

| id | target_0 | target_1 | target_2 | target_3 | target_4 | target_5 | target_6 |
|----|----------|----------|----------|----------|----------|----------|----------|
| 1  | 0.1      | 0.1      | 0.1      | 0.1      | 0.3      | 0.1      | 0.2      |
| 2  | 0.2      | 0.2      | 0.1      | 0.1      | 0.1      | 0.1      | 0.2      |
| .. | ...      | ...      | ...      | ...      | ...      | ...      | ...      |
| N  | p_{N1}   | p_{N2}   | p_{N3}   | p_{N4}   | p_{N5}   | p_{N6}   | p_{N7}   |

## 1. 实验总体目标

围绕你的数据（question, answer, label ∈ {0…6}），设计一套实验来回答三个主要问题：

1. **Q1：不同 HPO 方法在相同资源预算下，谁更有效？**

   * 对比 Grid Search / Random Search / Optuna TPE 在多分类 logloss 上的表现与收敛速度。

2. **Q2：特征和模型的选择对最终效果的影响有多大？**

   * 对比 TF-IDF 特征 vs 句向量特征（sentence embedding），以及线性模型 vs 树模型。

3. **Q3：在训练资源有限的前提下，最合适的一套 pipeline 是什么？**

   * 选出「性能不错 + 训练快 + 实现简单」的组合，作为最终提交和 report 的主方法。

---

## 2. 数据划分与评估协议

### 2.1 数据结构

你的原始 CSV 大致为：

* `question`：字符串
* `answer`：字符串
* `label`：整数 0–6（对应 7 个 LLM）

Kaggle 评估指标是 7 类的 **multiclass logloss**，提交需要输出每一类的概率。

### 2.2 数据划分策略

目标：既要有本地稳定评估，又要能提交 Kaggle。

建议：

1. **本地训练集 / 本地 hold-out 测试集**

   * 从训练数据中按 question 做 group split，划分 80% 为训练 + CV，20% 作为本地 hold-out。
   * group 可以通过对 `question` 文本做 hash 得到 `question_id`，保证同一 question 不会同时出现在 train/val/test。

2. **训练内部使用 GroupKFold 做 CV**

   * K=5，`GroupKFold(n_splits=5)`，group 为 question_id。
   * 每个 trial 计算 5 折平均 logloss，作为目标函数。

3. **Kaggle 提交**

   * 最后用在训练集上表现最好的配置，在全量训练集上重训一个模型，对 Kaggle test 预测，提交 logloss 作为参考（放到 report 中）。

### 2.3 评价指标

* 主指标：

  * 多分类 logloss（与比赛完全对齐）
* 辅助指标：

  * Accuracy、macro F1（用于 report 辅助说明）

每个实验都至少记录：

* mean CV logloss ± std
* 本地 hold-out logloss
* （可选）Kaggle public LB logloss

---

## 3. 特征工程设计

为了「实现简单 + 训练快」，特征计划分两类：

### 3.1 文本预处理（统一）

对每条样本：

* 拼接文本：

  * `text = "[Q] " + question + " [A] " + answer`
* 统一预处理（在 Pipeline 里做）：

  * lower case 与否作为一个可调选项（可以在 HPO 里让 trial 决定）
  * 去掉多余空白，保留标点

### 3.2 特征方案 A：TF-IDF（主力）

* word-level TF-IDF：

  * `ngram_range` ∈ {(1,1), (1,2), (1,3)}
  * `max_features` ∈ {20000, 40000, 60000}
* char-level TF-IDF：

  * `ngram_range` ∈ {(3,5), (3,6)}
  * `max_features` ∈ {20000, 40000}

组合方式：

* A1：仅使用 word TF-IDF
* A2：仅使用 char TF-IDF
* A3：word + char 拼接（sklearn `FeatureUnion`）

在 HPO 时可以用一个离散超参数选择 A1/A2/A3。

### 3.3 特征方案 B：句向量（sentence embeddings）

* 选一个轻量 sentence-transformers 模型，例如 `all-MiniLM-L6-v2`（384 维）
* 对 `text` 做编码，得到 384 维向量（或 question / answer 各 384 维，拼成 768 维）
* 不微调，仅做前向；**离线预计算并缓存**所有样本的 embedding（.npy）。

这是一个很适合写在 report 里的对照组：

* TF-IDF 属于稀疏高维词袋；
* sentence embedding 属于密集低维语义特征。

### 3.4 轻量统计特征（可选）

额外添加一些简单数值特征，用于解释：

* 文本长度（字符、token 数）
* 标点比例、数字比例、大写比例
* 平均词长

你可以做一个只用统计特征 + Tree 模型的小实验，或者和 TF-IDF / embedding 拼接，主要用于分析「不同 LLM 的风格差异」。

---

## 4. 模型与 HPO 对象

为了控制实验复杂度，模型家族限定为两类：

### 4.1 模型家族 1：线性模型（Logistic Regression）

* 适用于：TF-IDF 特征
* 超参数空间（示例）：

  * `C`：log-uniform [1e-3, 1e2]
  * `penalty`：['l2']（若使用 saga，可加入 'l1'）
  * `class_weight`：['none', 'balanced']
  * `max_iter`：统一设一个较大值，如 200 / 500，防止不收敛

优点：训练非常快，是非常好的 HPO 对象。

### 4.2 模型家族 2：梯度提升树（LightGBM 或 XGBoost）

建议 LightGBM，多分类接口较方便。

* 适用于：

  * sentence embeddings
  * TF-IDF 压缩后的低维特征（如果需要，可用 TruncatedSVD 降维）
* 典型超参数空间：

  * `num_leaves`：[16, 255]
  * `max_depth`：[-1, 4, 6, 8, 10]
  * `learning_rate`：log-uniform [1e-3, 0.3]
  * `n_estimators`：[100, 1000]（可做整数 uniform 或 log-uniform）
  * `min_data_in_leaf`：[10, 200]
  * `feature_fraction`：[0.6, 1.0]
  * `bagging_fraction`：[0.6, 1.0]
  * `lambda_l1`, `lambda_l2`：log-uniform [1e-4, 10]

相较线性模型训练稍慢，但在低维 dense 特征上仍然可以接受。

---

## 5. HPO 方法与设置

你要比较至少 3 种 HPO 方法：

### 5.1 方法一：Grid Search

* 用于：**线性模型 + 单一 TF-IDF 配置**，作为经典 baseline。

* 搜索空间尽量小，保证可跑完，比如：

  * `C` ∈ {0.01, 0.1, 1.0, 10.0}
  * `ngram_range` ∈ {(1,1), (1,2)}
  * `class_weight` ∈ {None, 'balanced'}

* 实现：`GridSearchCV` + `GroupKFold(5)` + `scoring='neg_log_loss'`。

* 记录：

  * 总 trial 数 = 参数组合数
  * 总耗时
  * best config & best CV logloss

### 5.2 方法二：Random Search

* 用于：同样的 pipeline（TF-IDF + LR），但参数用分布采样。

* 搜索空间：

  * `C`：log-uniform [1e-3, 1e2]
  * `ngram_range`：离散集合 {(1,1), (1,2), (1,3)}
  * `max_features`：{20000, 40000, 60000}
  * `class_weight`：{None, 'balanced'}

* 预算设置：

  * `n_iter = 30` 或 `50`（保证总评估次数 ≈ Grid Search）

* 实现：`RandomizedSearchCV` + 同样的 `GroupKFold(5)`。

### 5.3 方法三：Optuna TPE（重点）

* 用于：更大的 search space，可以把**特征选择 + 模型选择 + 超参**全部放进去。

* 目标函数 `objective(trial)`：

  1. 先让 trial 决定：

     * 特征类型：TF-IDF（A1/A2/A3） or sentence embedding
     * 模型类型：LR vs LightGBM
  2. 根据特征 / 模型类型，进一步采样对应超参数
  3. 构建完整 pipeline
  4. 做 GroupKFold(5) CV，返回 mean logloss

* Sampler：`TPESampler()`（默认即可）

* Pruner：`MedianPruner()` 或 `SuccessiveHalvingPruner()`（在 CV fold 层面剪枝，减少无效 trial 耗时）

* 预算设置：

  * 小预算：`n_trials = 30`
  * 中预算：`n_trials = 60`
  * 大预算：`n_trials = 100`

记录：

* 每个 trial 的配置、CV logloss、训练时间
* best trial 的配置和性能
* 「trial index vs best-so-far logloss」收敛曲线（后面画图用）

---

## 6. 实验分组与对比设计

下面是一套可以直接照着跑的实验矩阵，每一条都是一个「实验组」。

### 6.1 组 A：Baseline 与数据 sanity check

**目的：** 确认 pipeline 正常，建立无 HPO 基线。

* A1：`TF-IDF(word(1,2), max_features=40000) + LR(C=1, class_weight=None)`

  * 无 HPO，直接用默认超参
  * 记录 CV / hold-out / Kaggle logloss
* A2：`TF-IDF(word+char) + LR`，手动稍微调一下参数

这部分在 report 的 Experiments 前半段写成「Baseline」，也可以顺带对比 StratifiedKFold vs GroupKFold 的差异（说明数据泄露问题）。

### 6.2 组 B：固定模型 + 三种 HPO 方法对比（主实验 1）

**固定 pipeline：**

* 特征：`TF-IDF(word+char)`，初始超参取一个合理默认（或者部分也加入 search）
* 模型：`Logistic Regression`

**实验：**

* B1：GridSearchCV（参数网格如 5.1）
* B2：RandomizedSearchCV，`n_iter` 调到与 B1 trial 数相近
* B3：Optuna TPE，`n_trials` 设置为与 B1/B2 总 trial 数接近（例如 40）

**对比维度：**

* best CV logloss（主）
* hold-out logloss
* 总耗时
* 参数分布 / 最优参数位置

**可视化：**

* 三种方法的 best logloss 条形图
* Random & TPE 的「trial index vs best-so-far logloss」曲线（展示收敛速度差异）

这一组是作业里最重要的对比之一。

### 6.3 组 C：更复杂模型下的 HPO 对比（主实验 2）

**固定 pipeline：**

* 特征：sentence embedding（MiniLM 预计算）
* 模型：LightGBM 多分类

**实验：**

* C1：GridSearchCV（小网格，比如 `num_leaves`×`max_depth`×`learning_rate` 组合）
* C2：RandomizedSearchCV，`n_iter=30`
* C3：Optuna TPE，`n_trials=60`（稍多一点，以体现 TPE 在高维空间的优势）

**分析重点：**

* 相比于线性模型，LightGBM 的超参数空间更大，Grid Search 更吃力；
* Random / TPE 的优势更明显；
* 度量每种方法在【同样时间预算】下的最优 logloss。

### 6.4 组 D：特征表示对比（主实验 3）

选择一种 HPO 方法（建议 Optuna），只比较特征：

* D1：TF-IDF（word only） + LR
* D2：TF-IDF（word+char） + LR
* D3：sentence embedding + LR
* D4：sentence embedding + LightGBM

都用 Optuna TPE，`n_trials=50`，保持公平。

**分析点：**

* 稀疏词袋 vs 密集语义向量在 LLM 作者识别上的优劣
* 特征维度、训练时间、性能之间的 trade-off

### 6.5 组 E：消融实验与错误分析（用于 report 深度）

1. **Q vs A 重要性**

   * E1：仅用 answer 文本
   * E2：仅用 question 文本
   * E3：question+answer 拼接（对比已有最佳模型）

2. **统计特征**

   * E4：统计特征 + LightGBM
   * E5：统计特征 + TF-IDF 拼接，对判别某些模型是否有帮助

3. **错误分析**

   * 用最终最佳模型，对各类混淆情况做分析：

     * 哪两类 LLM 容易互相混淆
     * 选一些典型错误样本做 case study（方便写在 Experiments/Discussion 里）

---

## 7. 最终模型与提交

* 根据上述实验，选出一个「性能–复杂度–训练时间」最均衡的配置，例如：

  * TF-IDF(word+char) + LR，Optuna TPE 调参后的最优 config
  * 或 sentence embedding + LightGBM
* 在整个训练集（不再划分 hold-out）上用该 config 重训模型
* 对 Kaggle test 输出 7 类概率，生成 submission.csv
* 记录 Kaggle public LB logloss，作为最终结果写入 report。

可以再尝试一个简单 ensemble（比如将两个最好模型的预测概率做平均），检查是否有进一步提升；如果有，也可以写进 paper 的 Experiments 部分。

---

## 8. 报告写作时可以重点强调的实验维度

* HPO 方法对比：

  * 性能：best CV logloss
  * 效率：每次评估平均时间、总时间
  * 收敛速度 / trial 利用效率

* search space 设计的重要性：

  * 展示 poor search space 会导致三种 HPO 都差，good space 三种方法整体好，但 TPE 收敛更快

* 模型复杂度：

  * 线性 vs Tree 的差异
  * 训练时间、内存占用

* 特征表示：

  * TF-IDF vs sentence embedding
  * 哪类 LLM 更依赖语义信息 / 表达风格信息

---

如果你需要下一步更具体一点，我可以帮你把「组 B：TF-IDF + LR + 三种 HPO」整理成一份伪代码/实验脚本结构（比如 `run_experiment_B.py` 里面怎么组织函数、参数和日志），这样你可以直接照着写代码。
