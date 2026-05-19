import argparse
import pickle
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.config import (
    FAKE_DATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TRUE_DATA_PATH,
    VECTORIZER_PATH,
)
from src.data_loader import add_clean_content, load_dataset
from src.evaluation import (
    evaluate_model,
    plot_model_comparison,
    plot_probability_distribution,
    plot_roc_curve,
)

warnings.filterwarnings("ignore")


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        stop_words="english",
    )


def print_top_features(model, vectorizer, count: int = 10) -> None:
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    top_real = np.argsort(coefficients)[-count:]
    top_fake = np.argsort(coefficients)[:count]

    print("\nTop words for REAL news:\n")
    for i in reversed(top_real):
        print(feature_names[i])

    print("\nTop words for FAKE news:\n")
    for i in top_fake:
        print(feature_names[i])


def save_artifacts(model, vectorizer) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(VECTORIZER_PATH, "wb") as file:
        pickle.dump(vectorizer, file)

    print("\nModel and vectorizer saved successfully.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Vectorizer path: {VECTORIZER_PATH}")


def train(show_plots: bool = True) -> None:
    df = load_dataset(FAKE_DATA_PATH, TRUE_DATA_PATH)
    print("Dataset Shape:", df.shape)

    df = add_clean_content(df)

    vectorizer = build_vectorizer()
    x = vectorizer.fit_transform(df["content"])
    y = df["label"]
    print("Feature matrix shape:", x.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)
    nb_pred = nb_model.predict(x_test)

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    param_grid = {"C": [0.1, 1, 5, 10]}

    grid = GridSearchCV(lr_model, param_grid, cv=5, scoring="f1")
    grid.fit(x_train, y_train)

    best_lr = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)
    lr_pred = best_lr.predict(x_test)

    nb_results = evaluate_model(y_test, nb_pred, "Naive Bayes", show=show_plots)
    lr_results = evaluate_model(y_test, lr_pred, "Logistic Regression", show=show_plots)

    cv_scores = cross_val_score(best_lr, x, y, cv=5, scoring="f1")
    print("Cross-validation F1 scores:", cv_scores)
    print("Average CV F1 score:", cv_scores.mean())

    print("\nClass distribution:")
    print(df["label"].value_counts())
    df["length"] = df["content"].apply(lambda text: len(text.split()))
    print("\nAverage length by class:")
    print(df.groupby("label")["length"].mean())

    plot_probability_distribution(best_lr, x_test, show=show_plots)
    plot_model_comparison(
        ["Naive Bayes", "Logistic Regression"],
        [nb_results[3], lr_results[3]],
        show=show_plots,
    )
    plot_roc_curve(best_lr, x_test, y_test, show=show_plots)

    print_top_features(best_lr, vectorizer)
    save_artifacts(best_lr, vectorizer)

    print("\nFINAL MODEL PERFORMANCE SUMMARY\n")
    print(f"Naive Bayes F1 Score: {nb_results[3]:.4f}")
    print(f"Logistic Regression F1 Score: {lr_results[3]:.4f}")
    print(f"Cross-validation F1 Score: {cv_scores.mean():.4f}")
    print("\nBest Model: Logistic Regression (Tuned)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the fake news classifier.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Train without opening matplotlib charts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(show_plots=not args.no_plots)
