import os
import numpy as np
import torch
from ultralytics import YOLO
from roboflow import Roboflow
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Initialize Roboflow with your API key
rf = Roboflow(api_key="Y0esjVkvzESVzRIDvaX7")
project = rf.workspace().project("ghostnetvimpublic")
dataset = project.version("1").download("yolov8")

# Define path to dataset
train_data_path = 'D:/0000000000000000 OOD NOVIONA/Test/ghostnetvimpublic.v1i.yolov8/train/images'
ood_data_path = 'D:/0000000000000000 OOD NOVIONA/Test/ghostnetvimpublic.v1i.yolov8/ood/images'
test_data_path = 'D:/0000000000000000 OOD NOVIONA/Test/ghostnetvimpublic.v1i.yolov8/valid/images'

print(f"Train data path: {train_data_path}")
print(f"OOD data path: {ood_data_path}")
print(f"Test data path: {test_data_path}")

# Verify paths exist
def check_path(path):
    if not os.path.exists(path):
        print(f"Error: The path {path} does not exist.")
        return False
    return True

train_path_exists = check_path(train_data_path)
ood_path_exists = check_path(ood_data_path)
test_path_exists = check_path(test_data_path)

if not (train_path_exists and ood_path_exists and test_path_exists):
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
if not check_images_exist(ood_data_path):
    print(f"Error: No valid image files found in {ood_data_path}")
if not check_images_exist(test_data_path):
    print(f"Error: No valid image files found in {test_data_path}")

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
ood_dataset = ObjectDetectionDataset(ood_data_path, transform=transform)
test_dataset = ObjectDetectionDataset(test_data_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the YOLOv8-Ghost model
model = YOLO("yolov8n-ghost-p6.yaml")

# Start training the model on your custom dataset
print("Start Training")
model.train(data="D:/0000000000000000 OOD NOVIONA/Test/ghostnetvimpublic.v1i.yolov8/data.yaml", epochs=50)

# Function to compute logit bias from OOD data
def compute_logit_bias(model, data_loader):
    model.eval()
    logits = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            batch_logits = [det['conf'] for det in outputs]
            logits.extend(batch_logits)
    logits = np.concatenate(logits, axis=0)
    logit_mean = np.mean(logits, axis=0)
    return logit_mean

# Calculate logit bias using OOD data
logit_bias = compute_logit_bias(model, ood_loader)

# Function to apply logit bias adjustment
def apply_vim(logits, logit_bias, scale_factor=1.0):
    return logits + scale_factor * logit_bias

# Function for inference with VIM adjustment
def inference_with_vim(model, inputs, logit_bias, scale_factor=1.0):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        adjusted_logits = apply_vim(outputs.logits.cpu().numpy(), logit_bias, scale_factor)
    return adjusted_logits


results = []
# Loop for epochs
for epoch in range(1, 11):
    print(f"Starting epoch {epoch}")
    train_results = model.train(data="D:/0000000000000000 OOD NOVIONA/Test/ghostnetvimpublic.v1i.yolov8/data.yaml", epochs=1)
    
    # Collect metrics from train_results
    metrics = {
        'epoch': epoch,
        'train/box_loss': train_results.box_loss,
        'train/cls_loss': train_results.cls_loss,
        'train/dfl_loss': train_results.dfl_loss,
        'metrics/precision(B)': train_results.precision,
        'metrics/recall(B)': train_results.recall,
        'metrics/mAP50(B)': train_results.map50,
        'metrics/mAP50-95(B)': train_results.map,
        'val/box_loss': train_results.val_box_loss,
        'val/cls_loss': train_results.val_cls_loss,
        'val/dfl_loss': train_results.val_dfl_loss,
        'lr/pg0': train_results.lr0,
        'lr/pg1': train_results.lr1,
        'lr/pg2': train_results.lr2
    }
    results.append(metrics)

# Save results to CSV
import csv

csv_file = 'results.csv'
csv_columns = [
    'epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
    'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
    'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 
    'lr/pg0', 'lr/pg1', 'lr/pg2'
]

try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results:
            writer.writerow(data)
except IOError:
    print("I/O error")

# Example inference with test data
for inputs, _ in test_loader:
    adjusted_logits = inference_with_vim(model, inputs, logit_bias)
    # Use adjusted_logits for further inference

# Evaluate model performance with adjusted logits and fine-tune logit bias if necessary
scale_factors = [0.5, 1.0, 1.5, 2.0]
for scale in scale_factors:
    for inputs, _ in test_loader:
        adjusted_logits = inference_with_vim(model, inputs, logit_bias, scale_factor=scale)