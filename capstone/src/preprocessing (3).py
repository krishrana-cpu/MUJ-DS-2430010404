import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
RAW_CSV_PATH = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
RAW_IMAGE_FOLDERS = [
    "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_1",
    "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/HAM10000_images_part_2",
]

OUTPUT_DATA_DIR = "/kaggle/working/data"
RESULTS_DIR = "/kaggle/working/results"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

LABEL_MAP = {
    "akiec": "actinic_keratoses",
    "bcc": "basal_cell_carcinoma",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic_nevi",
    "vasc": "vascular_lesions",
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Load and label the metadata
# ---------------------------------------------------------
print("Reading metadata CSV...")
df = pd.read_csv(RAW_CSV_PATH)
df["label"] = df["dx"].map(LABEL_MAP)

print(f"Total labeled images: {len(df)}")
print(f"Classes found: {df['label'].nunique()}")

# ---------------------------------------------------------
# STEP 2: 70/15/15 train/val/test split (stratified by class)
# ---------------------------------------------------------
train_df, temp_df = train_test_split(
    df, test_size=(VAL_RATIO + TEST_RATIO), stratify=df["label"], random_state=RANDOM_SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
    stratify=temp_df["label"], random_state=RANDOM_SEED
)

# ---------------------------------------------------------
# STEP 3: Save the dataset distribution table
# ---------------------------------------------------------
distribution = pd.DataFrame({
    "train": train_df["label"].value_counts(),
    "val": val_df["label"].value_counts(),
    "test": test_df["label"].value_counts(),
}).fillna(0).astype(int)
distribution["total"] = distribution.sum(axis=1)
distribution.loc["TOTAL"] = distribution.sum()

distribution_path = os.path.join(RESULTS_DIR, "dataset_distribution.csv")
distribution.to_csv(distribution_path)

# ---------------------------------------------------------
# STEP 4: Index raw image files by image_id
# ---------------------------------------------------------
print("\nIndexing image files...")
image_path_lookup = {}
for folder in RAW_IMAGE_FOLDERS:
    if not os.path.isdir(folder):
        print(f"WARNING: folder '{folder}' not found.")
        continue
    for filename in os.listdir(folder):
        image_id = os.path.splitext(filename)[0]
        image_path_lookup[image_id] = os.path.join(folder, filename)
print(f"Indexed {len(image_path_lookup)} files.")

# ---------------------------------------------------------
# STEP 5: Copy images into data/train, data/val, data/test
# ---------------------------------------------------------
def sort_images(dataframe, split_name):
    copied, skipped = 0, 0
    for _, row in dataframe.iterrows():
        source_path = image_path_lookup.get(row["image_id"])
        if source_path is None:
            skipped += 1
            continue
        dest_folder = os.path.join(OUTPUT_DATA_DIR, split_name, row["label"])
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy2(source_path, os.path.join(dest_folder, os.path.basename(source_path)))
        copied += 1
    print(f"{split_name}: copied {copied}, skipped {skipped}")

print("\nSorting images into data/train, data/val, data/test...")
sort_images(train_df, "train")
sort_images(val_df, "val")
sort_images(test_df, "test")
print("\nDone. Next: run train.py")
