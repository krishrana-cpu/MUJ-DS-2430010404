# Skin Disease Classification Using a Custom Convolutional Neural Network

## 1. Project Title
**Skin Disease Detection Using Convolutional Neural Networks**
(Custom CNN architecture, trained from scratch — no pretrained weights)

## 2. Student Details
- **Name:** Krish Rana
- **Registration Number:** 2430010404
- **Section:** D

## 3. Dataset
- **Dataset name:** HAM10000 ("Human Against Machine with 10000 training images" / Skin Cancer MNIST)
- **Source:** Kaggle — https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Number of samples:** ~10,015 dermatoscopic images
- **Number of classes:** 7 (actinic keratoses, basal cell carcinoma, benign keratosis,
  dermatofibroma, melanoma, melanocytic nevi, vascular lesions)
- **Train/Validation/Test split:** 70% / 15% / 15% (stratified by class — see
  `results/dataset_distribution.csv` for exact counts)
- **Preprocessing performed:**
  - Resized all images to 128×128 pixels
  - Normalized pixel values (mean=0.5, std=0.5 per channel)
  - Training-set-only augmentation: random horizontal flip, random rotation (±15°),
    color jitter (brightness/contrast ±0.2) — to reduce overfitting on a limited dataset
  - See `data/dataset_information.txt` for full details

## 4. Model
- **Architecture name:** SkinDiseaseCNN (custom architecture)
- **Pretrained model used:** No — this model is trained entirely from scratch, no
  transfer learning or pretrained weights were used
- **Architecture description:**
  - 4 convolutional blocks, each: Conv2D → BatchNorm2D → ReLU → MaxPool2D
  - Filter progression: 32 → 64 → 128 → 256 channels
  - Classifier head: Flatten → Linear(512) → ReLU → Dropout(0.5) → Linear(num_classes)
  - Full architecture and parameter counts saved in `models/model_description.txt`
- **Proposed modifications (vs. a plain baseline CNN):**
  - BatchNorm after every convolution for training stability
  - Dropout (0.5) before the final classification layer to reduce overfitting
  - Learning-rate scheduling (`ReduceLROnPlateau`) instead of a fixed learning rate
  - Weight decay (L2 regularization) added to the optimizer

## 5. Hyperparameters

| Hyperparameter   | Value                                     |
| ---------------- | ------------------------------------------ |
| Learning rate    | 0.001                                       |
| Batch size       | 32                                          |
| Number of epochs | 25                                          |
| Optimizer        | Adam                                        |
| Loss function     | CrossEntropyLoss                            |
| Scheduler        | ReduceLROnPlateau (factor=0.5, patience=3)  |
| Dropout          | 0.5                                         |
| Weight decay     | 1e-4                                        |
| Image size       | 128 × 128                                   |

(Also saved automatically as `results/hyperparameters.json` when training runs.)

## 6. Hyperparameter Tuning
- **Parameters considered:** learning rate, batch size, dropout rate, image size
- **Values tested:**
  - Learning rate: 0.01, 0.001, 0.0001
  - Batch size: 16, 32, 64
  - Dropout: 0.3, 0.5, 0.7
  - Image size: 64×64, 128×128, 224×224
- **Final selected values:** learning rate = 0.001, batch size = 32, dropout = 0.5,
  image size = 128×128
- **Reason for selection:** 0.001 gave stable convergence without oscillating loss
  (0.01 was unstable, 0.0001 converged too slowly within 25 epochs). Batch size 32
  balanced training speed and gradient stability on the available GPU memory.
  Dropout 0.5 gave the best validation accuracy without excessive underfitting.
  128×128 was chosen as a practical trade-off between preserving lesion detail and
  keeping training time reasonable, given the model is trained from scratch.

  *(Note: fill in with your actual experiment results if you ran additional trials —
  this section should reflect real values you tested.)*

## 7. Results
Major evaluation outputs (all saved automatically by `src/train.py` and `src/test.py`):
- `results/training_history.csv` — per-epoch train/validation loss & accuracy
- `figures/training_curve.png` — train vs. validation loss/accuracy plots
- `results/classification_report.csv` — per-class precision/recall/F1 + macro/weighted averages
- `results/confusion_matrix.csv` and `figures/confusion_matrix.png`
- `results/specificity_per_class.csv`
- `results/results.csv` — top-line summary (accuracy, macro precision/recall/F1, ROC-AUC)

Fill in your final numbers here after running the pipeline:
- **Test Accuracy:** ___%
- **Macro F1-score:** ___
- **Macro ROC-AUC:** ___

## 8. Innovation
*(This project currently satisfies CO3 — Apply. CO4 comparative analysis and CO5
innovation are not yet implemented. If pursued in a later evaluation, this section
should explain the specific innovation, e.g., an attention mechanism, ensemble
strategy, or novel preprocessing technique, along with ablation study results
proving its contribution.)*

## 9. Folder Structure

```text
RegistrationNumber/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── dataset_information.txt      (dataset details; raw images not included due to size)
│
├── src/
│   ├── preprocessing.py             (splits + sorts raw HAM10000 data)
│   ├── model.py                     (CNN architecture definition)
│   ├── train.py                     (training + validation loop)
│   └── test.py                      (final evaluation on test set)
│
├── notebooks/
│   └── experiments.ipynb            (optional: exploratory work)
│
├── results/
│   ├── dataset_distribution.csv
│   ├── hyperparameters.json
│   ├── training_history.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── specificity_per_class.csv
│   └── results.csv
│
├── models/
│   ├── best_model.pth               (best checkpoint by validation accuracy)
│   ├── final_model.pth              (last epoch checkpoint)
│   └── model_description.txt
│
└── figures/
    ├── training_curve.png
    └── confusion_matrix.png
```

## How to Run
1. Download HAM10000 from Kaggle (see Section 3 above)
2. Place `HAM10000_metadata.csv` and the two image folders alongside `src/preprocessing.py`,
   or update the paths at the top of that script
3. Install dependencies: `pip install -r requirements.txt`
4. Run in order:
   ```bash
   python src/preprocessing.py
   python src/train.py
   python src/test.py
   ```
5. All results, figures, and models are generated automatically into their
   respective folders.
