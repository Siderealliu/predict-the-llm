import sys
import pandas as pd
from pathlib import Path

sys.path.append(".")

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config.data_config import get_default_data_config
from src.config.model_config import LR_BASE_PARAMS
from src.data.data_loader import load_data, basic_preprocessing
from src.data.preprocessor import TextPreprocessor
from src.features.statistical_features import StatisticalFeatures


# Optimal parameter
BEST_PARAMS = {
    "ngram_range": [1, 3],
    "max_features": 40000,
    "C": 0.14445251022763064,
    "class_weight": "balanced"
}

def main():
    
    paths, _ = get_default_data_config()
    print(f"reading data...\n   Train: {paths.train_path}\n   Test:  {paths.test_path}")
    
    train_df, test_df = load_data(paths)
    
    train_df = basic_preprocessing(train_df)
    test_df["Question"] = test_df["Question"].fillna("")
    test_df["Response"] = test_df["Response"].fillna("")
    
    print(f"   Train size: {len(train_df)}")
    print(f"   Test size:  {len(test_df)}")

    # Prepare the TF-IDF parameters
    tfidf_params = {
        "ngram_range": tuple(BEST_PARAMS["ngram_range"]), 
        "max_features": BEST_PARAMS["max_features"],
        "min_df": 2
    }
    
    # Prepare the LR parameters
    lr_params = LR_BASE_PARAMS.copy()
    lr_params.update({
        "C": BEST_PARAMS["C"],
        "class_weight": BEST_PARAMS["class_weight"]
    })

    pipeline = Pipeline([
        (
            "features",
            FeatureUnion([
                ("tfidf", Pipeline([
                    ("prep", TextPreprocessor(mode="qa")), 
                    ("vec", TfidfVectorizer(**tfidf_params))
                ])),
                ("stat", StatisticalFeatures()),
            ])
        ),
        ("classifier", LogisticRegression(**lr_params))
    ])


    pipeline.fit(train_df, train_df["target"])


    # predict
    preds_proba = pipeline.predict_proba(test_df)
    
    submission = pd.DataFrame(
        preds_proba, 
        columns=[f"target_{i}" for i in range(7)]
    )
    
    submission.insert(0, "id", test_df["id"])
    
    output_file = "submission.csv"
    submission.to_csv(output_file, index=False)
    
    print(f"save to: {output_file}")
    print(submission.head())

if __name__ == "__main__":
    main()
