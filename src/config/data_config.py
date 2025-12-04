from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
    """集中管理数据路径，方便统一修改。"""

    base_dir: Path = Path("data/h2oai-predict-the-llm")
    train_filename: str = "train.csv"
    test_filename: str = "test.csv"
    sample_submission_filename: str = "sample_submission.csv"

    @property
    def train_path(self) -> Path:
        return self.base_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        return self.base_dir / self.test_filename

    @property
    def sample_submission_path(self) -> Path:
        return self.base_dir / self.sample_submission_filename


@dataclass
class SplitConfig:
    """数据划分相关配置。"""

    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    group_col: str = "question_hash"


def get_default_data_config() -> tuple[DataPaths, SplitConfig]:
    """返回默认数据路径与划分配置。"""

    return DataPaths(), SplitConfig()
