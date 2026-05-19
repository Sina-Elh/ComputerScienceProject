import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.config import OUTPUTS_DIR


def _save_plot(filename: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {path}")


def _finish_plot(show: bool) -> None:
    if show:
        plt.show()
    else:
        plt.close()


def evaluate_model(y_test, y_pred, model_name: str, show: bool = True):
    print(f"\n===== {model_name} =====")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    safe_name = model_name.lower().replace(" ", "_")
    _save_plot(f"{safe_name}_confusion_matrix.png")
    _finish_plot(show)

    return acc, prec, rec, f1


def plot_probability_distribution(model, x_test, show: bool = True):
    plt.figure()
    sns.histplot(model.predict_proba(x_test)[:, 1], bins=50)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Probability of Real News")
    _save_plot("prediction_probability_distribution.png")
    _finish_plot(show)


def plot_model_comparison(model_names, f1_scores, show: bool = True):
    plt.figure()
    plt.bar(model_names, f1_scores)
    plt.title("Model Comparison (F1 Score)")
    plt.ylabel("F1 Score")
    _save_plot("model_comparison_f1_score.png")
    _finish_plot(show)


def plot_roc_curve(model, x_test, y_test, show: bool = True):
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    _save_plot("roc_curve_logistic_regression.png")
    _finish_plot(show)
