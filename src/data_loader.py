import pandas as pd

from src.config import FAKE_DATA_PATH, RANDOM_STATE, TRUE_DATA_PATH
from src.preprocessing import prepare_content


def load_dataset(fake_path=FAKE_DATA_PATH, true_path=TRUE_DATA_PATH) -> pd.DataFrame:
    missing_files = [path for path in (fake_path, true_path) if not path.exists()]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Dataset files were not found. Put Fake.csv and True.csv in the data folder:\n"
            f"{missing}"
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def add_clean_content(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"title", "text"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["content"] = df.apply(
        lambda row: prepare_content(row["title"], row["text"]),
        axis=1,
    )
    return df
