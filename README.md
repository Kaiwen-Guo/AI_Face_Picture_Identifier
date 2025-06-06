# AI Face Picture Identifier

A simple project to classify AI-generated faces vs. real human faces.

## Folder structure
- `src/` contains Python modules used for training and evaluation.
- `notebooks/` contains the original Colab notebooks.
- Pretrained weights (`*.npy`) are stored in the repository root.

## Usage
Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Prepare your dataset so that `data_root` has two subfolders: `ai_faces/` and `real_faces/` with images inside. Then run:

```bash
python -m src.train data_root --model logreg --img-size 64 --pca 50 --C 1.0 --output models
```

This will train either a logistic regression or linear SVM model and save it to `models/model.joblib`.

## Datasets
- [AI generated faces](https://www.kaggle.com/datasets/chelove4draste/ai-generated-faces)
- [LFW people](https://www.kaggle.com/datasets/atulanandjha/lfwpeople)

