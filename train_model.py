import os
import numpy as np
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Define path to dataset
train_data_path = 'D:/Test/ghostnetwithv8mergedextraimages.v1i.yolov8/train/images'
test_data_path = 'D:/Test/ghostnetwithv8mergedextraimages.v1i.yolov8/test/images'
valid_data_path = 'D:/Test/ghostnetwithv8mergedextraimages.v1i.yolov8/valid/images'

print(f"Train data path: {train_data_path}")
print(f"OOD data path: {test_data_path}")
print(f"Test data path: {valid_data_path}")

# Verify paths exist
def check_path(path):
    if not os.path.exists(path):
        print(f"Error: The path {path} does not exist.")
        return False
    return True

train_path_exists = check_path(train_data_path)
test_path_exists = check_path(test_data_path)
valid_path_exists = check_path(valid_data_path)

if not (train_path_exists and test_path_exists and valid_path_exists):
    raise FileNotFoundError("One or more data paths do not exist. Please check your paths.")

# Function to check if directories contain images
def check_images_exist(path):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    for root, _, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                return True
    return False

if not check_images_exist(train_data_path):
    print(f"Error: No valid image files found in {train_data_path}")
if not check_images_exist(test_data_path):
    print(f"Error: No valid image files found in {test_data_path}")
if not check_images_exist(valid_data_path):
    print(f"Error: No valid image files found in {valid_data_path}")

# Define a custom dataset for object detection
class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # returning img_path for debugging purposes

# Load datasets with transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ObjectDetectionDataset(train_data_path, transform=transform)
test_dataset = ObjectDetectionDataset(test_data_path, transform=transform)
valid_dataset = ObjectDetectionDataset(valid_data_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

# Load the YOLOv8-Ghost model
model = YOLO(model="novi.yaml")

# Start training the model on your custom dataset for 50 epochs
print("Start Training")
model.train(data="D:/Test/ghostnetwithv8mergedextraimages.v1i.yolov8/data.yaml", epochs=100)

# After training, try to retrieve the checkpoint manually
try:
    model.ckpt = torch.load("runs/detect/train10/weights/best.pt")
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Failed to load checkpoint: {e}")

# Check the contents of self.ckpt before saving
if model.ckpt is None:
    raise ValueError("Error: model.ckpt is None. Ensure the model has been trained properly.")
else:
    print("model.ckpt is populated.")

# Save the trained model
model_save_path = "D:/Test/runs/detect/train22/trained_model.pt"
model.save(model_save_path)

print(f"Model saved to {model_save_path}")