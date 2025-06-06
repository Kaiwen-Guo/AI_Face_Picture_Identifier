from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


def train_logistic_regression(X: np.ndarray, y: np.ndarray, penalty: str | None = 'l2', C: float = 1.0) -> LogisticRegression:
    model = LogisticRegression(penalty=penalty if penalty else 'none', C=C, solver='liblinear' if penalty else 'lbfgs', max_iter=1000)
    model.fit(X, y)
    return model


def train_svm(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> LinearSVC:
    svm = LinearSVC(C=C)
    svm.fit(X, y)
    return svm


def apply_pca(X_train: np.ndarray, X_val: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_val_pca, pca


def evaluate(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return accuracy_score(y, preds)

