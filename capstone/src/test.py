# =========================================================
# test.py
# =========================================================
# Evaluates the trained model on the held-out TEST set (never seen
# during training or validation). Produces all the metrics/tables
# required by the assignment's "Model Performance" section:
#   - Confusion matrix
#   - Accuracy, Precision, Recall, F1 (per-class + macro/weighted)
#   - ROC-AUC (one-vs-rest, multi-class)
#
# Run AFTER src/train.py has produced models/best_model.pth
# =========================================================

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score, precision_recall_fscore_support
)

sys.path.append(os.path.dirname(__file__))
from model import SkinDiseaseCNN

DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
FIGURES_DIR = "figures"
IMAGE_SIZE = 128
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/test", transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# ---------------------------------------------------------
# LOAD TRAINED MODEL (best checkpoint, by validation accuracy)
# ---------------------------------------------------------
model = SkinDiseaseCNN(num_classes=num_classes, image_size=IMAGE_SIZE).to(device)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth"), map_location=device))
model.eval()
print("Loaded best_model.pth")

# ---------------------------------------------------------
# RUN INFERENCE ON TEST SET
# ---------------------------------------------------------
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ---------------------------------------------------------
# OVERALL ACCURACY
# ---------------------------------------------------------
test_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# ---------------------------------------------------------
# PER-CLASS + MACRO/WEIGHTED PRECISION, RECALL, F1
# ---------------------------------------------------------
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
print("\nClassification report:")
print(report_df)
print("Saved results/classification_report.csv")

# ---------------------------------------------------------
# ROC-AUC (multi-class, one-vs-rest)
# ---------------------------------------------------------
try:
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    print(f"\nMacro ROC-AUC (one-vs-rest): {roc_auc:.4f}")
except ValueError as e:
    roc_auc = None
    print(f"\nCould not compute ROC-AUC: {e}")

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"))
plt.show()
print("Saved figures/confusion_matrix.png and results/confusion_matrix.csv")

# ---------------------------------------------------------
# SPECIFICITY (per class, computed from confusion matrix)
# ---------------------------------------------------------
specificity_per_class = {}
for i, class_name in enumerate(class_names):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_per_class[class_name] = specificity

specificity_df = pd.DataFrame.from_dict(specificity_per_class, orient="index", columns=["specificity"])
specificity_df.to_csv(os.path.join(RESULTS_DIR, "specificity_per_class.csv"))
print("\nPer-class specificity:")
print(specificity_df)

# ---------------------------------------------------------
# SAVE SUMMARY RESULTS (top-line numbers for README/report)
# ---------------------------------------------------------
summary = {
    "test_accuracy": test_accuracy,
    "macro_precision": report_dict["macro avg"]["precision"],
    "macro_recall": report_dict["macro avg"]["recall"],
    "macro_f1": report_dict["macro avg"]["f1-score"],
    "weighted_f1": report_dict["weighted avg"]["f1-score"],
    "macro_roc_auc": roc_auc,
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
print("\nSaved results/results.csv with summary metrics:")
print(summary_df)

print("\nEvaluation complete. All results saved in results/ and figures/.")
