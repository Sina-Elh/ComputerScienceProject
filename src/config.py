from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FAKE_DATA_PATH = DATA_DIR / "Fake.csv"
TRUE_DATA_PATH = DATA_DIR / "True.csv"

MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.25

BIAS_WORDS = ["reuters", "said", "washington", "tuesday", "wednesday"]
