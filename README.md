# Hybrid-Ensemble-Learning-for-Autism-Spectrum-Disorder-Detection-Using-Autoencoders-and-Classifiers

This repository contains a small computer vision workflow for agricultural image analysis. It includes:

- A labeled image dataset in YOLO annotation format
- A dataset splitting script for train/validation/test preparation
- A PyTorch training script based on MobileNetV2

The project appears to be aimed at distinguishing between `crop` and `weed` classes from field images.
The dataset used in this project is from Kaggle.

## Project Structure

```text
pro/
+-- classes.txt
+-- datasplit.py
+-- train.py
+-- yo.py
+-- data/
|   +-- *.jpeg
|   `-- *.txt
`-- output_dataset/
    +-- train/
    |   +-- images/
    |   `-- labels/
    +-- val/
    |   +-- images/
    |   `-- labels/
    `-- test/
        +-- images/
        `-- labels/
```

## Dataset

- `data/` contains the original dataset files.
- The dataset source for this project is Kaggle.
- Each image has a matching `.txt` annotation file in YOLO format.
- `classes.txt` defines two classes:
  - `crop`
  - `weed`

Current dataset counts in this repo:

- Raw dataset: `1300` images and `1300` label files
- Train split: `1105` images / `1105` labels
- Validation split: `130` images / `130` labels
- Test split: `65` images / `65` labels

Example YOLO label:

```text
1 0.608398 0.498047 0.541016 0.531250
```

This follows the standard YOLO format:

```text
class_id x_center y_center width height
```

All box values are normalized to image width and height.

## Files

### `datasplit.py`

Splits the dataset into training, validation, and test sets by copying image files and their matching annotation files.

Current split ratios in the script:

- Train: `85%`
- Validation: `10%`
- Test: `5%`

Note:

- The script uses hardcoded Windows paths.
- The script currently points to `C:\Users\Pranav\Desktop\pro\...`, so the paths should be updated before running it in a different location.
- This repo already contains a prepared split under `output_dataset/`.

### `train.py`

Trains a MobileNetV2-based model in PyTorch with:

- Image resizing to `512 x 512`
- Data augmentation
- Initial feature freezing
- Fine-tuning after initial training
- Validation metrics
- Confusion matrix and classification report

Libraries used:

- `torch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `numpy`

Important limitation:

- `train.py` uses `torchvision.datasets.ImageFolder`, which expects images to be arranged in class-specific subfolders such as `train/crop/` and `train/weed/`.
- The current `output_dataset/train/images`, `output_dataset/val/images`, and `output_dataset/test/images` folders are flat and contain image files directly.
- Because of that, `train.py` will need path and dataset-structure changes before it can run successfully on the current split output.

### `yo.py`

This file is currently empty and does not affect the project.

## Requirements

Install the main dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

## How To Use

### 1. Check the dataset

Make sure each image in `data/` has a matching `.txt` annotation file.

### 2. Update script paths

Before running the scripts, update the hardcoded paths in:

- `datasplit.py`
- `train.py`

### 3. Split the dataset

```bash
python datasplit.py
```

### 4. Prepare the training layout

If you want to use `train.py` as written, reorganize the dataset into class folders compatible with `ImageFolder`, for example:

```text
train/
+-- crop/
`-- weed/
```

If your goal is object detection with YOLO annotations, you will likely want a detection training pipeline instead of the current classification script.

### 5. Train the model

```bash
python train.py
```

## Current Observations

- The repository already includes a YOLO-style annotated dataset and a completed split in `output_dataset/`.
- The annotations suggest this project started as an object detection dataset.
- The current PyTorch training script is written like an image classification pipeline, so it does not directly match the current dataset layout.

## Suggested Improvements

- Replace hardcoded absolute paths with relative paths
- Add a `requirements.txt`
- Decide whether the project is for:
  - object detection using YOLO, or
  - image classification using PyTorch
- Update the training pipeline so it matches the dataset format
- Add model saving, logging, and evaluation outputs to a dedicated results folder

## License

No license file is currently included in this repository.
