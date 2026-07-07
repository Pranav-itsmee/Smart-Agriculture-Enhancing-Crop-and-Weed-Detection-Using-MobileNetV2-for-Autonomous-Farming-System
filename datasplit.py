import os
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

# Define paths
dataset_dir = "C:\\Users\\Pranav\\Desktop\\pro\\"
images_dir = os.path.join(dataset_dir, "data")
labels_dir = os.path.join(dataset_dir, "data")

# Output directories
output_dir = "C:\\Users\\Pranav\\Desktop\\pro\\data"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Create output folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)

os.makedirs(val_dir, exist_ok=True)
os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

# Get all image file names
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle files
random.shuffle(image_files)

# Split dataset
train_split = 0.85  # 70% for training
val_split = 0.1    # 20% for validation
test_split = 0.05   # 10% for testing

train_count = int(len(image_files) * train_split)
val_count = int(len(image_files) * val_split)

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Function to copy files
def copy_files(file_list, dest_dir):
    for file in file_list:
        # Copy image
        shutil.copy(os.path.join(images_dir, file), os.path.join(dest_dir, "images", file))
        
        # Copy corresponding label file
        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(dest_dir, "labels", label_file))

# Copy files to respective directories
copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print("Dataset split complete.")
print(f"Train: {len(train_files)} images")
print(f"Validation: {len(val_files)} images")
print(f"Test: {len(test_files)} images")
