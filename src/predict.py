import argparse
import pickle

from src.config import MODEL_PATH, VECTORIZER_PATH
from src.preprocessing import preprocess_text, remove_bias_words


def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


def predict_news(text: str, threshold: float = 0.4) -> str:
    model, vectorizer = load_artifacts()

    processed = preprocess_text(text)
    processed = remove_bias_words(processed)
    vectorized = vectorizer.transform([processed])
    probability = model.predict_proba(vectorized)[0][1]

    if probability >= threshold:
        label = "Real News"
    else:
        label = "Fake News"

    return f"{label} (Confidence: {probability:.2f})"


def parse_args():
    parser = argparse.ArgumentParser(description="Predict whether a news text is real or fake.")
    parser.add_argument("text", help="News title or article text to classify.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Probability threshold for classifying text as real news.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(predict_news(args.text, threshold=args.threshold))
