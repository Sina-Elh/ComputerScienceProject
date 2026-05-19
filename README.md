# Fake News Detection Project

This project trains a machine learning model to classify news articles as fake or real.

## Project Structure

```text
.
├── data/
│   ├── Fake.csv              # Add this file manually
│   └── True.csv              # Add this file manually
├── models/
│   ├── fake_news_model.pkl   # Generated after training
│   └── tfidf_vectorizer.pkl  # Generated after training
├── outputs/
│   └── *.png                 # Generated charts after training
├── src/
│   ├── config.py             # Project paths and constants
│   ├── data_loader.py        # Dataset loading and content preparation
│   ├── evaluation.py         # Metrics and plots
│   ├── predict.py            # Prediction script
│   ├── preprocessing.py      # Text cleaning functions
│   └── train.py              # Main training script
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Notebook-to-File Mapping

| Notebook section | New file |
| --- | --- |
| All required libraries | `requirements.txt` and imports inside each Python file |
| Upload dataset | Not needed in PyCharm; place CSV files in `data/` |
| Load the uploaded dataset | `src/data_loader.py` |
| Cleaning and preprocessing | `src/preprocessing.py` |
| Apply preprocessing | `src/data_loader.py` |
| Feature engineering with TF-IDF | `src/train.py` |
| Train-test split | `src/train.py` |
| Naive Bayes and Logistic Regression training | `src/train.py` |
| Evaluation function and plots | `src/evaluation.py` |
| Cross-validation and model summary | `src/train.py` |
| Top features | `src/train.py` |
| Model saving and loading | `src/train.py` and `src/predict.py` |
| Final prediction function | `src/predict.py` |

## Requirements

- Python 3.10 or newer
- PyCharm
- The dataset files:
  - `Fake.csv`
  - `True.csv`

The code expects both CSV files to contain at least these columns:

- `title`
- `text`

## Setup in PyCharm

1. Open this folder as a PyCharm project.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a folder named `data` if it does not already exist.
5. Put the dataset files inside `data/`:

```text
data/Fake.csv
data/True.csv
```

## Train the Model

In PyCharm, run:

```bash
main.py
```

Or run this command from the project root:

```bash
python -m src.train
```

To train without opening chart windows:

```bash
python -m src.train --no-plots
```

After training, the model files are saved in `models/`:

```text
models/fake_news_model.pkl
models/tfidf_vectorizer.pkl
```

The charts are saved in `outputs/`:

```text
outputs/naive_bayes_confusion_matrix.png
outputs/logistic_regression_confusion_matrix.png
outputs/prediction_probability_distribution.png
outputs/model_comparison_f1_score.png
outputs/roc_curve_logistic_regression.png
```

## Make a Prediction

After training, run:

```bash
python -m src.predict "Government launches new healthcare policy to improve hospitals"
```

Example output:

```text
Real News (Confidence: 0.72)
```

