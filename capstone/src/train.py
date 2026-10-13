# =========================================================
# train.py
# =========================================================
# Trains the custom CNN using TRAIN data, evaluating on VALIDATION
# data after every epoch. This lets us plot training-vs-validation
# curves and check for overfitting -- required by the assignment's
# "Training Analysis" section.
#
# Run AFTER src/preprocessing.py has created data/train, data/val, data/test.
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from model import SkinDiseaseCNN

# ---------------------------------------------------------
# CONFIG / HYPERPARAMETERS
# ---------------------------------------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
FIGURES_DIR = "figures"

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SIZE = 128
OPTIMIZER_NAME = "Adam"
LOSS_FUNCTION = "CrossEntropyLoss"
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Save hyperparameters immediately, so they're documented regardless
# of how training goes (required for the README hyperparameter table)
hyperparams = {
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": OPTIMIZER_NAME,
    "loss_function": LOSS_FUNCTION,
    "scheduler": "ReduceLROnPlateau (factor=0.5, patience=3)",
    "dropout": DROPOUT,
    "weight_decay": WEIGHT_DECAY,
    "image_size": IMAGE_SIZE,
}
with open(os.path.join(RESULTS_DIR, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=2)
print("Saved hyperparameters.json")

# ---------------------------------------------------------
# DATA TRANSFORMS
# ---------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Validation/test data is NOT augmented -- we want to measure
# performance on realistic, unaltered images
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ---------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------
train_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/val", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# ---------------------------------------------------------
# MODEL, LOSS, OPTIMIZER
# ---------------------------------------------------------
model = SkinDiseaseCNN(num_classes=num_classes, image_size=IMAGE_SIZE).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ---------------------------------------------------------
# TRAIN + VALIDATE LOOP
# ---------------------------------------------------------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0

print("\nStarting training...\n")
for epoch in range(EPOCHS):
    # ---- Training phase ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # ---- Validation phase ----
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    # Save the best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")

# Always also save the final-epoch model
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "final_model.pth"))

# ---------------------------------------------------------
# SAVE TRAINING HISTORY (for report + reproducibility)
# ---------------------------------------------------------
history_df = pd.DataFrame(history)
history_df.index.name = "epoch"
history_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"))
print("Saved training_history.csv")

# ---------------------------------------------------------
# PLOT TRAIN VS VALIDATION CURVES
# ---------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Loss: Train vs Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.title("Accuracy: Train vs Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "training_curve.png"))
plt.show()
print("Saved figures/training_curve.png")

# Save model architecture summary (for models/model_description.txt)
with open(os.path.join(MODELS_DIR, "model_description.txt"), "w") as f:
    f.write("Model: SkinDiseaseCNN (custom, built from scratch -- no pretrained weights)\n\n")
    f.write(str(model))
    f.write(f"\n\nTotal parameters: {total_params:,}\n")
    f.write(f"Trainable parameters: {trainable_params:,}\n")
    f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
print("Saved models/model_description.txt")

print("\nNext: run src/test.py to evaluate on the held-out test set.")
