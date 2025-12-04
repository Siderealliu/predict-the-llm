# Predict-the-LLM 超参数优化实验完整实现方案

## 📖 项目概述

### 目标
围绕H2O.ai的"预测LLM输出来源"竞赛数据，设计系统化实验对比不同超参数优化(HPO)方法、特征工程策略和模型架构的效果。

### 核心研究问题
1. **Q1**: 不同HPO方法在相同资源预算下，谁更有效？
   - 对比Grid Search、Random Search、Optuna TPE在多分类logloss上的表现与收敛速度

2. **Q2**: 特征和模型的选择对最终效果的影响有多大？
   - 对比TF-IDF特征 vs 句向量特征，以及线性模型 vs 树模型

3. **Q3**: 在训练资源有限的前提下，最合适的一套pipeline是什么？
   - 选出「性能不错 + 训练快 + 实现简单」的组合

### 数据情况
- **训练集**: 23,527条样本 (Question, Response, target∈{0-6})
- **测试集**: 6,008条样本
- **评估指标**: 7类multiclass logloss
- **提交格式**: 每类概率的CSV文件

## 🏗️ 完整项目目录结构

```
predict-the-llm/
├── README.md
├── pyproject.toml
├── .gitignore
├── main.py
├── Outline.md                    # 本文件：完整实现方案
├── data/
│   └── h2oai-predict-the-llm/
│       ├── train.csv (23,527条)
│       ├── test.csv (6,008条)
│       └── sample_submission.csv
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── data_config.py        # 数据路径、划分策略配置
│   │   ├── model_config.py       # 超参数搜索空间定义
│   │   └── experiment_config.py  # 5个实验组的具体配置
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # CSV数据加载与基础清洗
│   │   ├── splitter.py           # GroupKFold与train/val划分
│   │   └── preprocessor.py       # 文本预处理(拼接、清洗)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py              # 特征提取器基类
│   │   ├── tfidf_features.py     # TF-IDF特征(word+char级别)
│   │   ├── embedding_features.py # 句向量特征(MiniLM预计算)
│   │   └── statistical_features.py # 统计特征(长度、比例等)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py        # 模型基类
│   │   ├── logistic_regression.py # LR封装
│   │   └── lightgbm_model.py    # LightGBM封装
│   ├── hpo/
│   │   ├── __init__.py
│   │   ├── base_hpo.py          # HPO基类
│   │   ├── grid_search.py       # GridSearchCV实现
│   │   ├── random_search.py     # RandomizedSearchCV实现
│   │   └── optuna_tpe.py        # Optuna TPE实现
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # 评估指标(logloss等)
│   │   └── validator.py         # 交叉验证器
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── group_a_baseline.py  # 组A: Baseline基线实验
│   │   ├── group_b_tfidf_lr.py  # 组B: TF-IDF+LR三种HPO对比
│   │   ├── group_c_embedding_lgb.py # 组C: Embedding+LightGBM三种HPO
│   │   ├── group_d_feature_comparison.py # 组D: 四种特征+模型组合对比
│   │   └── group_e_ablation.py  # 组E: 消融实验(Q vs A, 统计特征等)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            # 统一日志系统
│   │   ├── io_utils.py          # 结果保存/加载(JSON, CSV)
│   │   └── visualization.py     # 收敛曲线、对比图可视化
│   └── pipeline/
│       ├── __init__.py
│       ├── base_pipeline.py     # Pipeline基类
│       ├── tfidf_pipeline.py    # TF-IDF+模型Pipeline
│       └── embedding_pipeline.py # Embedding+模型Pipeline
├── results/
│   ├── experiments/             # 实验组结果存储
│   │   ├── group_a/
│   │   ├── group_b/
│   │   ├── group_c/
│   │   ├── group_d/
│   │   └── group_e/
│   ├── checkpoints/            # 训练过程模型检查点
│   ├── cache/                  # Embedding等中间结果缓存
│   │   └── embeddings/
│   │       └── train_embeddings.npy
│   └── final/                  # 最终结果
│       ├── best_model.pkl      # 最佳模型
│       ├── submission.csv      # Kaggle提交文件
│       └── experiment_summary.csv # 所有实验对比表
├── logs/
│   ├── experiment.log          # 主实验日志
│   └── hpo_*.log               # 各HPO方法详细日志
├── scripts/
│   ├── run_all_experiments.py  # 批量运行所有实验
│   ├── run_single_experiment.py # 运行单个实验
│   └── prepare_submission.py   # 生成最终提交文件
└── venv/                       # 虚拟环境
```

## 📊 5个实验组详细设计方案

### 实验组A: Baseline与数据sanity check
**目标**: 建立基线，确认pipeline正常，验证GroupKFold的重要性

#### A1: 基础TF-IDF+LR
- **特征**: TF-IDF(word-level, ngram_range=(1,2), max_features=40000)
- **模型**: LogisticRegression(C=1.0, class_weight=None)
- **流程**: 直接训练，无HPO
- **记录**: CV logloss、holdout logloss、Kaggle logloss
- **意义**: 验证最简单的pipeline能否正常工作

#### A2: 增强TF-IDF+LR
- **特征**: TF-IDF(word+char组合, 使用FeatureUnion)
- **模型**: LR(C=0.1, class_weight='balanced')
- **意义**: 稍作调参，验证组合特征的效果

#### 数据泄露验证(可选)
- **对比**: GroupKFold vs StratifiedKFold
- **预期**: StratifiedKFold会有虚高的CV分数（因为同一question可能在不同fold）
- **意义**: 强调GroupKFold的必要性

### 实验组B: 固定TF-IDF+LR，三种HPO方法对比
**目标**: 在简单场景下对比三种HPO方法的效率与效果

#### 固定配置
- **特征**: TF-IDF(word+char组合, 默认参数)
- **模型**: LogisticRegression
- **搜索空间**: C、ngram_range、max_features、class_weight

#### B1: GridSearchCV
```python
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_features': [20000, 40000],
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__class_weight': [None, 'balanced']
}
```
- **试验数**: 2×2×4×2 = 32次
- **预期**: 穷举搜索，但计算量大

#### B2: RandomizedSearchCV
```python
param_dist = {
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
    'tfidf__max_features': [20000, 40000, 60000],
    'classifier__C': loguniform(1e-3, 1e2),
    'classifier__class_weight': [None, 'balanced']
}
```
- **试验数**: 30-40次（与Grid相近）
- **预期**: 在相同预算下覆盖更多参数空间

#### B3: Optuna TPE
```python
def objective(trial):
    ngram_range = trial.suggest_categorical('ngram_range', [(1,1), (1,2), (1,3)])
    max_features = trial.suggest_int('max_features', 20000, 60000, step=20000)
    C = trial.suggest_float('C', 1e-3, 1e2, log=True)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    # ...构建pipeline并评估
    return cv_score
```
- **试验数**: 40次
- **预期**: 相比随机搜索，TPE会更快收敛到优解

#### 对比维度
1. **性能**: best CV logloss
2. **效率**: 平均每trial时间
3. **收敛**: trial_index vs best_so_far曲线
4. **稳定性**: 不同随机种子下结果方差

### 实验组C: 固定Sentence Embedding+LightGBM，三种HPO对比
**目标**: 在复杂场景下验证HPO方法的扩展性

#### 固定配置
- **特征**: Sentence Embedding(all-MiniLM-L6-v2, 384维)
- **模型**: LightGBM多分类
- **预处理**: 离线预计算embedding并缓存

#### 嵌入预计算
```python
# data/embedding_features.py
class EmbeddingPreprocessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.cache_path = 'results/cache/embeddings/'

    def compute_and_cache(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(os.path.join(self.cache_path, 'train_embeddings.npy'), embeddings)
        return embeddings

    def load_cache(self):
        return np.load(os.path.join(self.cache_path, 'train_embeddings.npy'))
```

#### C1: GridSearchCV
```python
param_grid = {
    'num_leaves': [31, 63, 127],
    'max_depth': [5, 7, -1],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [200, 500]
}
```
- **试验数**: 3×3×3×2 = 54次
- **问题**: 高维空间下网格搜索效率极低

#### C2: RandomizedSearchCV
```python
param_dist = {
    'num_leaves': randint(16, 255),
    'max_depth': [-1, 4, 6, 8, 10],
    'learning_rate': loguniform(1e-3, 0.3),
    'n_estimators': randint(100, 1000),
    'min_data_in_leaf': randint(10, 200),
    'feature_fraction': uniform(0.6, 0.4),
    'lambda_l1': loguniform(1e-4, 10),
    'lambda_l2': loguniform(1e-4, 10)
}
```
- **试验数**: 30-50次
- **预期**: 相比Grid更高效

#### C3: Optuna TPE
```python
def objective(trial):
    num_leaves = trial.suggest_int('num_leaves', 16, 255)
    max_depth = trial.suggest_categorical('max_depth', [-1, 4, 6, 8, 10])
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
    # ...更多参数
    # 使用早期停止减少训练时间
    params = {
        'objective': 'multiclass',
        'num_class': 7,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    # CV评估...
    return cv_score
```
- **试验数**: 60次（给TPE更多空间展现优势）
- **预期**: TPE在高维空间优势更明显

#### 分析重点
1. **计算复杂度**: LightGBM训练时间 vs LR
2. **搜索效率**: TPE在复杂空间的优势
3. **剪枝效果**: MedianPruner减少无效trial

### 实验组D: 统一Optuna对比四种特征+模型组合
**目标**: 公平对比不同特征表示和模型的组合

#### 统一设置
- **HPO方法**: Optuna TPE (n_trials=50)
- **CV策略**: GroupKFold(n_splits=5)
- **指标**: 同样的logloss评估

#### 四种组合

**D1: TF-IDF(word only) + LR**
```python
feature_space = {
    'type': 'fixed',  # 特征类型固定为TF-IDF word
    'ngram_range': trial.suggest_categorical('ngram_range', [(1,1), (1,2), (1,3)]),
    'max_features': trial.suggest_int('max_features', 20000, 60000, step=20000)
}
model_space = {
    'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
}
```

**D2: TF-IDF(word+char) + LR**
```python
feature_space = {
    'type': 'tfidf_union',
    'word_ngram': trial.suggest_categorical('word_ngram', [(1,1), (1,2), (1,3)]),
    'char_ngram': trial.suggest_categorical('char_ngram', [(3,5), (3,6)]),
    'max_features': trial.suggest_int('max_features', 20000, 60000, step=20000)
}
```

**D3: Sentence Embedding + LR**
```python
feature_space = {
    'type': 'embedding',
    'model_name': 'all-MiniLM-L6-v2',  # 固定
    'dimension': 384  # 固定
}
model_space = {
    'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
}
```

**D4: Sentence Embedding + LightGBM**
```python
feature_space = {
    'type': 'embedding',
    'model_name': 'all-MiniLM-L6-v2'
}
model_space = {
    'num_leaves': trial.suggest_int('num_leaves', 16, 255),
    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    # ...更多LightGBM参数
}
```

#### 对比维度
1. **性能**: D1-D4的best CV logloss排序
2. **效率**: 平均每trial时间
3. **特征维度**: 稀疏高维 vs 密集低维
4. **训练速度**: LR vs LightGBM
5. **内存占用**: 不同特征的内存需求

#### 预期结论
- **D1 vs D2**: char-level TF-IDF是否带来提升
- **D2 vs D3**: 稀疏词袋 vs 密集语义向量
- **D3 vs D4**: 线性 vs 树模型在embedding上的表现

### 实验组E: 消融实验与深度分析
**目标**: 深入理解数据特性，识别关键特征

#### E1-E3: Q vs A重要性分析
**问题**: LLM风格主要体现在question还是answer中？

```python
# E1: 仅用question文本
text = '[Q] ' + df['Question']

# E2: 仅用answer文本
text = '[A] ' + df['Response']

# E3: question+answer拼接(标准做法)
text = '[Q] ' + df['Question'] + ' [A] ' + df['Response']
```
- **统一配置**: TF-IDF+LR, Optuna (n_trials=30)
- **分析**: 比较三种设置的best CV logloss
- **意义**: 了解哪部分文本更有判别性

#### E4: 统计特征 + LightGBM
**问题**: 简单的统计特征能否捕捉LLM风格差异？

```python
class StatisticalFeatures:
    def extract(self, text):
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'avg_word_len': np.mean([len(w) for w in text.split()]),
            'punct_ratio': sum(c in '.,!?;:' for c in text) / len(text),
            'upper_ratio': sum(c.isupper() for c in text) / len(text),
            'digit_ratio': sum(c.isdigit() for c in text) / len(text),
            'sentence_count': text.count('.') + text.count('!') + text.count('?')
        }
        return np.array(list(features.values()))
```
- **模型**: LightGBM (适合数值特征)
- **分析**: 统计特征能到多少logloss
- **意义**: 评估"风格"特征的可解释性

#### E5: 统计特征 + TF-IDF拼接
**问题**: 统计特征能否补充TF-IDF？

```python
# ColumnTransformer组合
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('tfidf', TfidfVectorizer(...), 0),  # 文本特征
        ('stat', StatisticalFeatures(), 1)   # 统计特征
    ])),
    ('classifier', LogisticRegression(...))
])
```
- **分析**: 拼接后logloss vs 纯TF-IDF的提升
- **意义**: 统计特征的增量价值

#### E6: 错误分析(可选深度)
**问题**: 哪些LLM容易被混淆？

```python
# 混淆矩阵分析
from sklearn.metrics import confusion_matrix

def analyze_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    # 可视化混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png')

    # 识别最容易混淆的类别对
    np.fill_diagonal(cm, 0)  # 去掉对角线
    confusion_pairs = []
    for i in range(7):
        for j in range(7):
            if cm[i, j] > 0:
                confusion_pairs.append((class_names[i], class_names[j], cm[i, j]))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    return confusion_pairs[:10]
```

**典型错误案例分析**:
- 选择几个被错误分类的样本
- 分析其question/response特点
- 讨论为什么会混淆

## 🔧 核心技术实现细节

### 1. 数据划分策略

#### GroupKFold实现
```python
# data/splitter.py
class GroupKFoldSplitter:
    def __init__(self, n_splits=5, group_col='question_hash', test_size=0.2):
        self.n_splits = n_splits
        self.group_col = group_col
        self.test_size = test_size

    def split(self, df):
        # 创建question hash作为group
        df['question_hash'] = df['Question'].apply(
            lambda x: abs(hash(x)) % 10000  # 0-9999的hash bucket
        )

        # 80/20划分
        train_df, holdout_df = train_test_split(
            df, test_size=self.test_size,
            stratify=df['target'],  # 保持label分布
            random_state=42
        )

        # 训练集上做5折GroupKFold
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, val_idx in gkf.split(
            train_df, train_df['target'], train_df['question_hash']
        ):
            yield train_idx, val_idx

        # holdout用于最终验证
        yield None, holdout_df.index  # 标记这是holdout
```

#### 使用示例
```python
splitter = GroupKFoldSplitter()
for fold, (train_idx, val_idx) in enumerate(splitter.split(df)):
    if train_idx is None:  # holdout
        X_holdout = df.loc[val_idx, ['Question', 'Response']]
        y_holdout = df.loc[val_idx, 'target']
    else:
        X_train = df.loc[train_idx, ['Question', 'Response']]
        y_train = df.loc[train_idx, 'target']
        X_val = df.loc[val_idx, ['Question', 'Response']]
        y_val = df.loc[val_idx, 'target']
```

### 2. 文本预处理

#### 统一预处理Pipeline
```python
# data/preprocessor.py
class TextPreprocessor:
    def __init__(self, lowercase=True, concatenate=True):
        self.lowercase = lowercase
        self.concatenate = concatenate

    def fit_transform(self, df):
        texts = []
        for _, row in df.iterrows():
            question = str(row['Question'])
            answer = str(row['Response'])

            if self.concatenate:
                text = f"[Q] {question} [A] {answer}"
            else:
                text = question + " " + answer

            if self.lowercase:
                text = text.lower()

            # 去除多余空白但保留标点
            text = re.sub(r'\s+', ' ', text).strip()
            texts.append(text)

        return texts

    def transform(self, df):
        return self.fit_transform(df)
```

### 3. 特征工程实现

#### TF-IDF组合特征
```python
# features/tfidf_features.py
class TfidfFeatureExtractor:
    def __init__(self, word_params, char_params, use_feature_union=True):
        self.word_params = word_params
        self.char_params = char_params
        self.use_feature_union = use_feature_union

    def build_pipeline(self):
        if self.use_feature_union:
            pipeline = Pipeline([
                ('preprocessor', TextPreprocessor()),
                ('features', FeatureUnion([
                    ('word_tfidf', TfidfVectorizer(**self.word_params)),
                    ('char_tfidf', TfidfVectorizer(**self.char_params))
                ])),
                ('classifier', LogisticRegression())
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', TextPreprocessor()),
                ('tfidf', TfidfVectorizer(**self.word_params)),
                ('classifier', LogisticRegression())
            ])
        return pipeline
```

#### 句向量特征
```python
# features/embedding_features.py
class EmbeddingFeatureExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='results/cache/embeddings'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def compute_embeddings(self, texts, force_recompute=False):
        cache_file = os.path.join(self.cache_dir, f'{self.model_name.replace("/", "_")}.npy')

        if not force_recompute and os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file)

        print(f"Computing embeddings with {self.model_name}...")
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        )

        np.save(cache_file, embeddings)
        print(f"Embeddings saved to {cache_file}")
        return embeddings

    def build_pipeline(self, lgb_params):
        # 先预计算embedding
        # 然后构建轻量pipeline
        pipeline = Pipeline([
            ('classifier', lgb.LGBMClassifier(**lgb_params))
        ])
        return pipeline
```

### 4. 模型封装

#### 统一模型接口
```python
# models/base_model.py
class BaseModel:
    def __init__(self, model_type, params):
        self.model_type = model_type
        self.params = params
        self.model = self._build_model()
        self.is_fitted = False

    def _build_model(self):
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.params.get('C', 1.0),
                class_weight=self.params.get('class_weight', None),
                max_iter=self.params.get('max_iter', 500),
                solver='liblinear' if self.params.get('penalty') == 'l2' else 'saga',
                penalty=self.params.get('penalty', 'l2'),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                objective='multiclass',
                num_class=7,
                metric='multi_logloss',
                boosting_type='gbdt',
                num_leaves=self.params.get('num_leaves', 31),
                max_depth=self.params.get('max_depth', -1),
                learning_rate=self.params.get('learning_rate', 0.1),
                n_estimators=self.params.get('n_estimators', 500),
                min_data_in_leaf=self.params.get('min_data_in_leaf', 20),
                feature_fraction=self.params.get('feature_fraction', 0.8),
                bagging_fraction=self.params.get('bagging_fraction', 0.8),
                lambda_l1=self.params.get('lambda_l1', 0),
                lambda_l2=self.params.get('lambda_l2', 0),
                verbose=-1,
                random_state=42,
                n_jobs=-1,
                force_col_wise=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
```

### 5. HPO方法实现

#### Grid Search封装
```python
# hpo/grid_search.py
class GridSearchHPO:
    def __init__(self, pipeline, param_grid, cv_strategy, scoring='neg_log_loss'):
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.cv_strategy = cv_strategy
        self.scoring = scoring

    def optimize(self, n_jobs=-1):
        print(f"Starting Grid Search with {len(ParameterGrid(self.param_grid))} combinations...")

        start_time = time.time()
        search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=self.cv_strategy,
            scoring=self.scoring,
            n_jobs=n_jobs,
            verbose=1,
            refit=True
        )
        search.fit(self.X_text, self.y)
        elapsed_time = time.time() - start_time

        results = {
            'method': 'grid_search',
            'n_trials': len(ParameterGrid(self.param_grid)),
            'total_time_seconds': elapsed_time,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'all_results': search.cv_results_,
            'refit_time': search.refit_time_
        }

        return results
```

#### Optuna TPE封装
```python
# hpo/optuna_tpe.py
class OptunaTPEHPO:
    def __init__(self, pipeline, param_space, cv_strategy, scoring='neg_log_loss'):
        self.pipeline = pipeline
        self.param_space = param_space
        self.cv_strategy = cv_strategy
        self.scoring = scoring
        self.trial_history = []

    def _sample_params(self, trial):
        params = {}
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")
        return params

    def _build_pipeline(self, params):
        # 根据params动态构建pipeline
        # 这里需要根据具体的param_space结构调整
        return self.pipeline  # 简化示意

    def optimize(self, n_trials, n_jobs=1, direction='maximize'):
        print(f"Starting Optuna TPE with {n_trials} trials...")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        def objective(trial):
            # 采样参数
            params = self._sample_params(trial)

            # 构建pipeline
            pipeline = self._build_pipeline(params)

            # CV评估
            start_time = time.time()
            scores = cross_val_score(
                pipeline, self.X_text, self.y,
                cv=self.cv_strategy,
                scoring=self.scoring,
                n_jobs=n_jobs
            )
            elapsed_time = time.time() - start_time

            # 记录trial历史
            self.trial_history.append({
                'trial_number': trial.number,
                'params': params,
                'cv_score': scores.mean(),
                'cv_std': scores.std(),
                'time_seconds': elapsed_time
            })

            return scores.mean()

        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        elapsed_time = time.time() - start_time

        # 收敛曲线
        best_scores = []
        current_best = -np.inf
        for trial in self.trial_history:
            if trial['cv_score'] > current_best:
                current_best = trial['cv_score']
            best_scores.append(current_best)

        results = {
            'method': 'optuna_tpe',
            'n_trials': n_trials,
            'total_time_seconds': elapsed_time,
            'best_params': study.best_params,
            'best_cv_score': study.best_value,
            'trial_history': self.trial_history,
            'convergence_curve': {
                'trial_index': list(range(n_trials)),
                'best_score_so_far': best_scores
            }
        }

        return results
```

### 6. 评估指标

#### 多分类LogLoss
```python
# evaluation/metrics.py
def compute_multiclass_logloss(y_true, y_pred_proba):
    """
    计算多分类logloss

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred_proba: 预测概率 (n_samples, n_classes)

    Returns:
        float: logloss值
    """
    return log_loss(y_true, y_pred_proba, labels=list(range(7)))

def evaluate_cv_results(y_true, y_pred_proba_list, cv_folds):
    """
    评估CV结果
    """
    scores = []
    for fold in range(len(cv_folds)):
        y_true_fold = [y_true[i] for i in cv_folds[fold]]
        y_pred_fold = y_pred_proba_list[fold]
        score = compute_multiclass_logloss(y_true_fold, y_pred_fold)
        scores.append(score)

    return {
        'mean_logloss': np.mean(scores),
        'std_logloss': np.std(scores),
        'fold_scores': scores
    }
```

## 📝 实验结果格式规范

### JSON结果格式
```json
{
  "experiment_metadata": {
    "experiment_id": "group_b_optuna_tpe",
    "experiment_group": "B",
    "timestamp": "2025-12-04T10:30:00",
    "python_version": "3.12.0",
    "libraries": {
      "scikit_learn": "1.3.0",
      "optuna": "3.5.0",
      "lightgbm": "4.1.0",
      "sentence_transformers": "2.2.2"
    }
  },
  "dataset_info": {
    "train_size": 23527,
    "n_classes": 7,
    "class_distribution": {
      "0": 3361,
      "1": 3361,
      "2": 3361,
      "3": 3361,
      "4": 3361,
      "5": 3361,
      "6": 3361
    },
    "split_strategy": "GroupKFold(n_splits=5)",
    "holdout_size": 4705
  },
  "experiment_config": {
    "feature_type": "tfidf_word_char",
    "model_type": "logistic_regression",
    "hpo_method": "optuna_tpe",
    "n_trials": 40,
    "cv_strategy": "GroupKFold(n_splits=5)",
    "scoring": "neg_log_loss"
  },
  "search_space": {
    "tfidf__word_ngram": {"type": "categorical", "choices": [[1,1], [1,2], [1,3]]},
    "tfidf__char_ngram": {"type": "categorical", "choices": [[3,5], [3,6]]},
    "tfidf__max_features": {"type": "int", "low": 20000, "high": 60000, "step": 20000},
    "classifier__C": {"type": "float", "low": 0.001, "high": 100, "log": true},
    "classifier__class_weight": {"type": "categorical", "choices": [null, "balanced"]}
  },
  "results": {
    "best_params": {
      "tfidf__word_ngram": [1, 2],
      "tfidf__char_ngram": [3, 5],
      "tfidf__max_features": 40000,
      "classifier__C": 0.854,
      "classifier__class_weight": "balanced"
    },
    "best_cv_score": -1.2345,
    "cv_scores_mean": -1.2456,
    "cv_scores_std": 0.0321,
    "cv_scores_fold": [-1.251, -1.238, -1.242, -1.249, -1.248],
    "holdout_score": -1.2298,
    "total_time_seconds": 3240.5,
    "per_trial_time_seconds": 81.0
  },
  "trial_history": [
    {
      "trial_id": 0,
      "params": {...},
      "cv_score": -1.5123,
      "cv_std": 0.0456,
      "time_seconds": 78.2
    },
    ...
  ],
  "convergence_curve": {
    "trial_index": [0, 1, 2, ..., 39],
    "best_score_so_far": [-1.512, -1.485, -1.463, ..., -1.234]
  },
  "resource_usage": {
    "peak_memory_mb": 2048,
    "avg_cpu_usage": 0.75
  }
}
```

### CSV对比表格格式
```csv
experiment_group,method,feature_type,model_type,n_trials,best_cv_score,best_cv_std,holdout_score,total_time_seconds,best_params_json,notes
A,baseline,tfidf_word,lr,0,-1.567,0.045,-1.543,120.5,"{C:1.0}","No HPO"
B,grid_search,tfidf_word_char,lr,32,-1.289,0.038,-1.265,2950.2,"{C:1.0, class_weight:balanced}",Exhaustive search
B,random_search,tfidf_word_char,lr,30,-1.276,0.035,-1.258,2234.1,"{C:0.854, class_weight:balanced}",Random sampling
B,optuna_tpe,tfidf_word_char,lr,40,-1.269,0.033,-1.251,2598.7,"{C:0.912, class_weight:balanced}",Bayesian optimization
C,grid_search,embedding,lgb,54,-1.198,0.029,-1.182,8520.3,"{num_leaves:63, lr:0.1}",Very slow
C,random_search,embedding,lgb,30,-1.187,0.031,-1.171,4750.8,"{num_leaves:95, lr:0.12}",Better efficiency
C,optuna_tpe,embedding,lgb,60,-1.175,0.028,-1.165,5680.4,"{num_leaves:127, lr:0.15}",Best performance
D,optuna_tpe,tfidf_word,lr,50,-1.298,0.036,-1.282,1650.0,"{C:1.2}","Word only"
D,optuna_tpe,tfidf_word_char,lr,50,-1.269,0.033,-1.251,2598.7,"{C:0.912}","Word+char"
D,optuna_tpe,embedding,lr,50,-1.256,0.034,-1.245,1820.3,"{C:0.785}","Embedding+LR"
D,optuna_tpe,embedding,lgb,50,-1.175,0.028,-1.165,5680.4,"{num_leaves:127}","Embedding+LGB"
```

## 📋 实现步骤

### 1. 项目基础搭建
**目标**: 建立项目骨架和基础依赖

**任务列表**:
1. **创建目录结构**
   ```bash
   mkdir -p src/{config,data,features,models,hpo,evaluation,experiments,utils,pipeline}
   mkdir -p results/{experiments/{group_a,group_b,group_c,group_d,group_e},cache/embeddings,final}
   mkdir -p logs
   mkdir -p scripts
   ```

2. **更新依赖包** (`pyproject.toml`)
   ```toml
   [project]
   name = "predict-the-llm"
   version = "0.1.0"
   requires-python = "==3.12"
   dependencies = [
       "numpy>=1.24.0",
       "pandas>=2.0.0",
       "scikit-learn>=1.3.0",
       "lightgbm>=4.1.0",
       "optuna>=3.5.0",
       "sentence-transformers>=2.2.2",
       "matplotlib>=3.7.0",
       "seaborn>=0.12.0",
       "tqdm>=4.65.0",
       "joblib>=1.3.0"
   ]
   ```
   安装依赖: `pip install -e .`

3. **创建基础配置文件**
   - `config/data_config.py`: 数据路径、划分策略
   - `config/model_config.py`: 超参数空间定义
   - `config/experiment_config.py`: 5个实验组配置

4. **实现基础工具**
   - `utils/logger.py`: 统一日志格式
   - `utils/io_utils.py`: JSON/CSV保存加载

**输出**:
- 完整的目录结构
- 配置文件模板
- 日志系统可用

### 2. 数据处理模块
**目标**: 实现数据加载、划分和预处理

**任务列表**:
1. **数据加载器** (`data/data_loader.py`)
   ```python
   def load_data(train_path, test_path):
       train_df = pd.read_csv(train_path)
       test_df = pd.read_csv(test_path)
       return train_df, test_df

   def basic_preprocessing(df):
       # 去除缺失值
       # 基础文本清洗
       return df
   ```

2. **GroupKFold划分器** (`data/splitter.py`)
   - 基于question hash创建group
   - 80/20 train/holdout split
   - 5折GroupKFold CV
   - 保持label分布平衡

3. **文本预处理器** (`data/preprocessor.py`)
   - 拼接question和answer
   - 可选lowercase
   - 去除多余空白

4. **集成测试**
   - 加载数据并检查基本统计信息
   - 验证划分是否保证同question不跨越fold
   - 保存预处理后的数据到cache

**输出**:
- 数据加载与划分函数
- 预处理器类
- 数据完整性验证脚本

### 3. 特征工程模块
**目标**: 实现三种特征提取器

**任务列表**:
1. **TF-IDF特征提取器** (`features/tfidf_features.py`)
   - word-level TF-IDF
   - char-level TF-IDF
   - FeatureUnion组合两者
   - 管道化处理(preprocessor + tfidf + classifier)

2. **句向量特征提取器** (`features/embedding_features.py`)
   - 集成sentence-transformers
   - all-MiniLM-L6-v2模型
   - 预计算并缓存到.npy
   - 支持增量计算(已有cache则跳过)

3. **统计特征提取器** (`features/statistical_features.py`)
   - 文本长度(字符、词数)
   - 标点比例、大写比例、数字比例
   - 平均词长、句子数
   - 可视化不同LLM的统计特征分布

4. **统一特征接口** (`features/base.py`)
   - `BaseFeatureExtractor`基类
   - `fit_transform()`标准接口
   - 支持序列化(pickle)

**输出**:
- 三种特征提取器
- 缓存机制
- 特征效果对比脚本

### 4. 模型模块
**目标**: 实现模型封装和Pipeline

**任务列表**:
1. **基础模型类** (`models/base_model.py`)
   - 统一接口(fit, predict, predict_proba)
   - LR和LightGBM的适配
   - 多分类配置(7类)

2. **Pipeline基类** (`pipeline/base_pipeline.py`)
   ```python
   class BasePipeline:
       def __init__(self, feature_extractor, model):
           self.feature_extractor = feature_extractor
           self.model = model

       def fit(self, X_text, y, cv_folds=None):
           # 特征提取
           X_features = self.feature_extractor.fit_transform(X_text)
           # 训练模型
           self.model.fit(X_features, y)
           return self

       def predict_proba(self, X_text):
           X_features = self.feature_extractor.transform(X_text)
           return self.model.predict_proba(X_features)
   ```

3. **TF-IDF Pipeline** (`pipeline/tfidf_pipeline.py`)
   - 集成预处理器 + TF-IDF + 模型
   - 支持网格搜索参数

4. **Embedding Pipeline** (`pipeline/embedding_pipeline.py`)
   - 加载预计算embedding + 模型
   - 轻量级pipeline

5. **集成测试**
   - 用小数据集验证pipeline正确性
   - 检查predict_proba输出形状(7类)

**输出**:
- 完整的模型和Pipeline类
- Pipeline正确性验证脚本

### 5. HPO模块
**目标**: 实现三种HPO方法

**任务列表**:
1. **Grid Search实现** (`hpo/grid_search.py`)
   - 封装GridSearchCV
   - 自动生成参数网格
   - 记录所有trial结果

2. **Random Search实现** (`hpo/random_search.py`)
   - 封装RandomizedSearchCV
   - 参数分布定义(loguniform, randint等)
   - 收敛曲线记录

3. **Optuna TPE实现** (`hpo/optuna_tpe.py`)
   - TPESampler配置(seed=42)
   - MedianPruner剪枝
   - 动态参数采样(根据trial选择)
   - Trial历史记录
   - 收敛曲线可视化

4. **统一HPO接口**
   ```python
   class HPOManager:
       def __init__(self, method, pipeline, param_space, cv_strategy):
           if method == 'grid':
               self.hpo = GridSearchHPO(pipeline, param_space, cv_strategy)
           elif method == 'random':
               self.hpo = RandomSearchHPO(pipeline, param_space, cv_strategy)
           elif method == 'optuna':
               self.hpo = OptunaTPEHPO(pipeline, param_space, cv_strategy)

       def optimize(self, budget):
           return self.hpo.optimize(budget)
   ```

5. **评估指标集成** (`evaluation/metrics.py`)
   - 多分类logloss计算
   - CV结果汇总
   - 辅助指标(Accuracy, F1)

**输出**:
- 三种HPO方法
- 统一调用接口
- 结果记录和可视化工具

### 6. 实验组实现
**目标**: 实现5个实验组的完整逻辑

**任务列表**:
1. **实验组A** (`experiments/group_a_baseline.py`)
   - A1: 基础TF-IDF+LR
   - A2: 增强TF-IDF+LR
   - GroupKFold vs StratifiedKFold对比(可选)
   - 输出基线性能

2. **实验组B** (`experiments/group_b_tfidf_lr.py`)
   - 固定TF-IDF+LR
   - 三种HPO方法(B1-B3)
   - 参数网格/分布定义
   - 收敛曲线对比

3. **实验组C** (`experiments/group_c_embedding_lgb.py`)
   - 固定Sentence Embedding+LightGBM
   - 预计算embedding缓存
   - 三种HPO方法对比
   - 计算资源监控

4. **实验组D** (`experiments/group_d_feature_comparison.py`)
   - D1-D4四种组合
   - 统一Optuna TPE (n_trials=50)
   - 性能、效率、维度对比
   - 雷达图可视化

5. **实验组E** (`experiments/group_e_ablation.py`)
   - E1-E3: Q vs A重要性
   - E4: 统计特征+LightGBM
   - E5: 统计特征+TF-IDF
   - E6: 错误分析(混淆矩阵)

**输出**:
- 5个实验组执行脚本
- 每个实验组的配置和结果格式

### 7. 实验执行与结果收集
**目标**: 运行所有实验并收集结果

**任务列表**:
1. **批量执行脚本** (`scripts/run_all_experiments.py`)
   ```python
   experiments = [
       ('A', 'baseline', None),
       ('B', 'grid', 32),
       ('B', 'random', 30),
       ('B', 'optuna', 40),
       ('C', 'grid', 54),
       ('C', 'random', 30),
       ('C', 'optuna', 60),
       ('D', 'optuna', 50),
       ('E', 'optuna', 30)
   ]

   for exp_group, method, budget in experiments:
       results = run_experiment(exp_group, method, budget)
       save_results(results)
   ```

2. **结果汇总**
   - 自动生成results/experiment_summary.csv
   - 合并所有实验的JSON结果
   - 识别最佳配置

3. **可视化脚本** (`utils/visualization.py`)
   - HPO收敛曲线对比
   - 不同HPO方法的条形图
   - 特征类型对比雷达图
   - 混淆矩阵热力图

4. **性能监控**
   - 记录每实验组的耗时
   - 内存使用监控
   - 生成资源使用报告

**输出**:
- 所有实验的完整结果
- 对比图表和报告
- 资源使用分析

### 8. 最终模型与提交
**目标**: 训练最佳模型并生成Kaggle提交

**任务列表**:
1. **选择最佳配置**
   ```python
   # 从实验结果中选择
   best_exp = find_best_experiment(results_summary_csv)
   print(f"Best config: {best_exp}")
   ```

2. **全量数据训练**
   ```python
   # 使用全部训练数据(不再划分holdout)
   final_pipeline = build_pipeline(best_exp.config)
   final_pipeline.fit(train_df['text'], train_df['target'])
   ```

3. **预测测试集**
   ```python
   test_probs = final_pipeline.predict_proba(test_df['text'])
   submission = pd.DataFrame({
       'id': test_df['id'],
       'target_0': test_probs[:, 0],
       'target_1': test_probs[:, 1],
       ...
       'target_6': test_probs[:, 6]
   })
   submission.to_csv('results/final/submission.csv', index=False)
   ```

4. **错误分析**
   - 混淆矩阵分析
   - 识别最容易混淆的LLM对
   - 选择3-5个典型错误案例
   - 分析为什么错误

5. **撰写总结报告**
   ```markdown
   ## 实验结论

   ### Q1: HPO方法对比
   - Optuna TPE在所有场景下均优于Grid和Random Search
   - 在高维空间(TPE+LightGBM)，优势更明显
   - 收敛速度: TPE > Random > Grid

   ### Q2: 特征与模型
   - Sentence Embedding + LightGBM 效果最佳
   - TF-IDF+LR 性价比最高(训练快+易实现+效果好)
   - 统计特征有一定帮助但提升有限

   ### Q3: 推荐配置
   - 资源充足: Sentence Embedding + LightGBM + Optuna TPE
   - 资源有限: TF-IDF(word+char) + LR + Optuna TPE
   ```

**输出**:
- final/submission.csv
- 错误分析报告
- 实验总结报告
- 最佳模型.pkl文件

## 🔍 关键实现细节

### 1. 统一日志系统
```python
# utils/logger.py
import logging
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# 使用示例
logger = setup_logger('experiment', f'logs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logger.info("Starting experiment group B...")
```

### 2. 结果保存与加载
```python
# utils/io_utils.py
import json
import pandas as pd
from datetime import datetime

def save_experiment_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"{timestamp}_results.json")

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {json_file}")
    return json_file

def load_experiment_results(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def create_summary_csv(results_dir, output_file):
    all_results = []
    for json_file in glob(os.path.join(results_dir, "*.json")):
        r = load_experiment_results(json_file)
        all_results.append({
            'experiment_group': r['experiment_config']['experiment_group'],
            'method': r['experiment_config']['hpo_method'],
            'best_cv_score': r['results']['best_cv_score'],
            'holdout_score': r['results']['holdout_score'],
            'total_time_seconds': r['results']['total_time_seconds']
        })

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"Summary CSV saved to {output_file}")
```

### 3. 并行化加速
```python
# Grid Search和Random Search
search = GridSearchCV(..., n_jobs=-1)  # 使用所有CPU核心

# Optuna
study.optimize(objective, n_jobs=4)  # 4个并行trial

# 交叉验证
scores = cross_val_score(..., cv=5, n_jobs=-1)
```

### 4. 剪枝策略
```python
# Optuna MedianPruner
pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
# 5折CV中，如果当前trial的分数低于前5折的中位数，则提前停止

# LightGBM早停
lgb.LGBMClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,
    eval_set=[(X_val, y_val)],
    verbose=-1
)
```

### 5. 内存优化
```python
# 分批处理大文本
def batch_encode(texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(batch)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

# 稀疏矩阵存储
from scipy.sparse import csr_matrix
tfidf_matrix = csr_matrix(tfidf_features)
```

## 📈 可视化方案

### 1. HPO收敛曲线
```python
def plot_convergence(results_list, save_path):
    plt.figure(figsize=(10, 6))
    for results in results_list:
        curve = results['convergence_curve']
        plt.plot(curve['trial_index'], curve['best_score_so_far'],
                label=f"{results['method']}")
    plt.xlabel('Trial Index')
    plt.ylabel('Best CV Score (higher is better)')
    plt.title('HPO Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
```

### 2. 性能对比条形图
```python
def plot_performance_comparison(summary_df, save_path):
    plt.figure(figsize=(12, 6))
    methods = summary_df['method'].unique()

    x = np.arange(len(methods))
    width = 0.35

    for i, group in enumerate(['B', 'C']):
        group_data = summary_df[summary_df['experiment_group'] == group]
        scores = [group_data[group_data['method'] == m]['best_cv_score'].values[0]
                 for m in methods if m in group_data['method'].values]
        plt.bar(x[:len(scores)] + i*width, scores, width,
               label=f'Group {group}', alpha=0.8)

    plt.xlabel('HPO Method')
    plt.ylabel('Best CV Score')
    plt.title('HPO Performance Comparison')
    plt.xticks(x + width/2, methods)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(save_path, dpi=300)
```

### 3. 特征类型雷达图
```python
def plot_feature_radar(summary_df, save_path):
    categories = ['Performance', 'Training Speed', 'Memory Efficiency', 'Implementation Complexity']
    D1 = [0.65, 0.95, 0.90, 0.95]  # TF-IDF+LR
    D2 = [0.72, 0.90, 0.85, 0.90]  # TF-IDF+char+LR
    D3 = [0.75, 0.70, 0.60, 0.75]  # Embedding+LR
    D4 = [0.88, 0.50, 0.55, 0.60]  # Embedding+LightGBM

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    D1 += D1[:1]
    D2 += D2[:1]
    D3 += D3[:1]
    D4 += D4[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, D1, 'o-', linewidth=2, label='D1: TF-IDF+LR')
    ax.fill(angles, D1, alpha=0.25)
    ax.plot(angles, D2, 'o-', linewidth=2, label='D2: TF-IDF+char+LR')
    ax.fill(angles, D2, alpha=0.25)
    ax.plot(angles, D3, 'o-', linewidth=2, label='D3: Embedding+LR')
    ax.fill(angles, D3, alpha=0.25)
    ax.plot(angles, D4, 'o-', linewidth=2, label='D4: Embedding+LightGBM')
    ax.fill(angles, D4, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Feature+Model Combination Comparison', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.savefig(save_path, dpi=300)
```

### 4. 混淆矩阵热力图
```python
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted LLM')
    plt.ylabel('True LLM')
    plt.title('Confusion Matrix - LLM Identification')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
```

## ⚠️ 潜在风险与解决方案

### 1. 内存不足
**风险**: Embedding预计算占用大量内存(23K×384≈9MB，可接受)

**解决方案**:
- 分批计算(每批64样本)
- 使用`np.save()`压缩存储
- 及时释放中间变量

### 2. 训练时间过长
**风险**: LightGBM在Large SearchSpace下训练慢

**解决方案**:
- 设置`early_stopping_rounds=50`
- 使用`Optuna MedianPruner`剪枝
- 限制`n_trials`在合理范围(30-60)

### 3. 数据泄露
**风险**: 同question在train和val中同时出现

**解决方案**:
- 必须使用`GroupKFold`
- 基于question text的hash划分
- 在A组实验中对比验证

### 4. 结果不稳定
**风险**: 随机种子不同导致结果差异

**解决方案**:
- 设置固定random_state
- 多次运行取平均
- 记录不同seed下的方差

### 5. 提交格式错误
**风险**: Kaggle要求严格格式

**解决方案**:
```python
# 验证提交格式
def validate_submission(submission_df):
    assert submission_df.shape[0] == 6008
    assert list(submission_df.columns) == ['id', 'target_0', ..., 'target_6']
    assert np.allclose(submission_df.iloc[:, 1:].sum(axis=1), 1.0)
    print("Submission format validated!")
```

## 🎯 成功指标

### 1. 实验完整性
- ✅ 所有5个实验组均成功运行
- ✅ 每个实验组的结果文件完整(JSON+CSV)
- ✅ 无数据泄露或格式错误

### 2. 性能指标
- **组B最佳CV logloss**: < 1.28
- **组C最佳CV logloss**: < 1.20
- **最终Kaggle logloss**: < 1.18 (目标)

### 3. 效率指标
- **组B三方法对比**: Optuna TPE收敛最快
- **组C三方法对比**: TPE在复杂空间优势明显
- **总体时间预算**: < 6小时(CPU训练)

### 4. 代码质量
- ✅ 模块化设计，可复用
- ✅ 统一接口，易扩展
- ✅ 完整注释和文档
- ✅ 结果格式统一，便于汇总

## 📚 扩展方向(可选)

### 1. 高级特征
- **N-gram TF-IDF**: 尝试(1,4), (1,5)
- **预训练词向量**: Word2Vec, GloVe
- **BERT Embedding**: 但计算成本高

### 2. 高级模型
- **XGBoost**: 与LightGBM对比
- **神经网络**: 轻量MLP
- **Ensemble**: 模型融合提升

### 3. 高级HPO
- **Optuna Sampler对比**: TPESampler vs CmaEsSampler vs NSGAIISampler
- **Pruner对比**: MedianPruner vs SuccessiveHalvingPruner
- **多目标优化**: 同时优化logloss和训练时间

### 4. 深度分析
- **SHAP**: 解释模型预测
- **LIME**: 单样本解释
- **Attention可视化**: 如果使用Transformer

---

## 📝 总结

本方案提供了完整的HPO实验实现蓝图，涵盖：

1. **系统化实验设计**: 5个实验组循序渐进
2. **模块化代码架构**: 高内聚低耦合，易维护
3. **统一结果格式**: JSON+CSV，便于对比分析
4. **详细实现步骤**: 可执行的开发指南
5. **风险控制机制**: 识别问题并提供解决方案

通过遵循本方案，你将能够：
- 系统对比不同HPO方法的效果
- 深入理解特征工程的影响
- 找出最适合资源约束的pipeline
- 生成高质量的实验报告

**下一步**: 确认计划后开始实施！
