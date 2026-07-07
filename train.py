import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
IMG_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device",DEVICE)
# Paths to dataset directories
train_dir = "C:\\Users\\Pranav\\Desktop\\pro\\output_dataset\\train\\images"
validation_dir = "C:\\Users\\Pranav\\Desktop\\pro\\output_dataset\\val\\images"

# Data Augmentation and Transformation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load Dataset
train_dataset = ImageFolder(train_dir, transform=train_transforms)
val_dataset = ImageFolder(validation_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Modify the classifier for binary classification
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.last_channel, 1),
    nn.Sigmoid()
)

model = model.to(DEVICE)

# Freeze the base model for initial training
for param in model.features.parameters():
    param.requires_grad = False

# Define Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and Validation Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).int()
            running_corrects += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.float().to(DEVICE)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).int()
                val_corrects += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    return history

history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Save the initial model
torch.save(model.state_dict(), "weed_crop_detector_mobilenetv2.pth")

# Unfreeze layers for fine-tuning
for param in model.features.parameters():
    param.requires_grad = True

# Lower learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)

# Fine-tune the Model
fine_tune_history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS // 2)

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_weed_crop_detector_mobilenetv2.pth")

# Evaluate the Model
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs).squeeze()
        preds = (outputs > 0.5).int()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Generate Classification Report and Confusion Matrix
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Weeds", "Crops"]))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Plot Training History
def plot_history(history, fine_tune_history=None):
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    if fine_tune_history:
        train_acc += fine_tune_history['train_acc']
        val_acc += fine_tune_history['val_acc']
        train_loss += fine_tune_history['train_loss']
        val_loss += fine_tune_history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss Over Epochs")

    plt.show()

plot_history(history, fine_tune_history)
