import os
from typing import Tuple
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


def load_images(folder: str, img_size: Tuple[int, int]) -> np.ndarray:
    """Load all images from a folder and return as an array of flattened grayscale pixels."""
    images = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            with Image.open(path) as im:
                im = im.convert("L").resize(img_size)
                images.append(np.asarray(im, dtype=np.float32).flatten())
        except Exception:
            continue
    return np.stack(images)


def load_dataset(ai_dir: str, real_dir: str, img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load AI generated and real faces into arrays."""
    X_ai = load_images(ai_dir, img_size)
    X_real = load_images(real_dir, img_size)
    y_ai = np.ones(len(X_ai), dtype=np.int64)
    y_real = np.zeros(len(X_real), dtype=np.int64)
    X = np.concatenate([X_ai, X_real], axis=0)
    y = np.concatenate([y_ai, y_real], axis=0)
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """Split dataset into train/validation/test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state, stratify=y)
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - relative_val_size, random_state=random_state, stratify=y_temp)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

