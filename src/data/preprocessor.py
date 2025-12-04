"""文本预处理：拼接Q/A，清洗空白。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class TextPreprocessor(BaseEstimator, TransformerMixin):
    lowercase: bool = True
    mode: str = "qa"  # qa | q | a
    prefix_question: str = "[Q]"
    prefix_answer: str = "[A]"

    def _combine(self, question: str, answer: str) -> str:
        if self.mode == "q":
            text = f"{self.prefix_question} {question}"
        elif self.mode == "a":
            text = f"{self.prefix_answer} {answer}"
        else:
            text = f"{self.prefix_question} {question} {self.prefix_answer} {answer}"
        return text

    def fit(self, df: pd.DataFrame, y=None):  # noqa: D401
        return self

    def transform(self, df: pd.DataFrame) -> List[str]:
        texts: List[str] = []
        for _, row in df.iterrows():
            question = str(row.get("Question", ""))
            answer = str(row.get("Response", ""))
            text = self._combine(question, answer)
            if self.lowercase:
                text = text.lower()
            text = re.sub(r"\s+", " ", text).strip()
            texts.append(text)
        return texts
