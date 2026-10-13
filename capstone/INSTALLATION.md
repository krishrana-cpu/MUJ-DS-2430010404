# Installation Guide

## Requirements
- Python 3.10+
- A GPU is strongly recommended (Google Colab free GPU works fine)

## Steps
1. Clone the repository
   ```bash
   git clone https://github.com/<GITHUB-USERNAME>/MUJ-DS-2430010404.git
   cd MUJ-DS-2430010404/capstone
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Download the HAM10000 dataset from Kaggle:
   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   Place `HAM10000_metadata.csv`, `HAM10000_images_part_1/` and `HAM10000_images_part_2/`
   inside the `capstone/` folder (or edit the paths at the top of `src/preprocessing.py`).
4. Run the pipeline in order
   ```bash
   python src/preprocessing.py
   python src/train.py
   python src/test.py
   ```
5. Outputs appear in `results/`, `models/` and `figures/`.
