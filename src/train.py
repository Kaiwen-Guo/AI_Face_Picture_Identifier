import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from . import data
from . import models


def main(data_root: str, img_size: int, use_pca: int | None, model: str, C: float, output: str):
    ai_dir = Path(data_root) / "ai_faces"
    real_dir = Path(data_root) / "real_faces"
    X, y = data.load_dataset(str(ai_dir), str(real_dir), (img_size, img_size))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data.split_data(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if use_pca:
        X_train, X_val, pca = models.apply_pca(X_train, X_val, use_pca)
        X_test = pca.transform(X_test)
    else:
        pca = None

    if model == "logreg":
        clf = models.train_logistic_regression(X_train, y_train, penalty='l2', C=C)
    else:
        clf = models.train_svm(X_train, y_train, C=C)

    val_acc = models.evaluate(clf, X_val, y_val)
    test_acc = models.evaluate(clf, X_test, y_test)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    Path(output).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "pca": pca}, Path(output) / "model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face identifier")
    parser.add_argument("data_root", help="Directory containing ai_faces/ and real_faces/")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--pca", type=int, default=None, help="Number of PCA components")
    parser.add_argument("--model", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    main(args.data_root, args.img_size, args.pca, args.model, args.C, args.output)

