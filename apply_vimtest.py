import numpy as np
import torch
from torch.utils.data import DataLoader
import csv
from ultralytics import YOLO
from torchvision import transforms
from train_model import ObjectDetectionDataset, check_images_exist

# Define path to dataset
valid_data_path = 'D:/Test/ghostnetwithv8mergedextraimages.v1i.yolov8/valid/images'
test_data_path = 'D:/Test/ghostnetwithv8mergedextraimages.v1i.yolov8/test/images'

# Load the YOLOv8-Ghost model
model = YOLO("D:/Test/runs/detect/train22/trained_model.pt")

# Load datasets with transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

valid_dataset = ObjectDetectionDataset(valid_data_path, transform=transform)
test_dataset = ObjectDetectionDataset(test_data_path, transform=transform)

# Create data loaders
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to compute logit bias from validation data
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

# Calculate logit bias using validation data
logit_bias = compute_logit_bias(model, valid_loader)

# Function to apply logit bias adjustment
def apply_vim(logits, logit_bias, scale_factor=1.0):
    return logits + scale_factor * logit_bias

# Function for inference with ViM adjustment
def inference_with_vim(model, inputs, logit_bias, scale_factor=1.0):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        adjusted_logits = apply_vim(outputs.logits.cpu().numpy(), logit_bias, scale_factor)
    return adjusted_logits

# Function to evaluate model performance
def evaluate_model(model, data_loader, logit_bias=None, scale_factor=1.0):
    model.eval()
    metrics = {'precision': [], 'recall': [], 'mAP50': [], 'mAP50-95': []}
    with torch.no_grad():
        for inputs, _ in data_loader:
            if logit_bias is not None:
                logits = inference_with_vim(model, inputs, logit_bias, scale_factor)
            else:
                logits = model(inputs).logits.cpu().numpy()
            # Here, you would calculate precision, recall, mAP50, and mAP50-95 based on the logits
            precision, recall, mAP50, mAP50_95 = calculate_metrics(logits)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['mAP50'].append(mAP50)
            metrics['mAP50-95'].append(mAP50_95)
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    return metrics
def calculate_metrics(logits):
    precision = 0.5
    recall = 0.5
    mAP50 = 0.5
    mAP50_95 = 0.5
    return precision, recall, mAP50, mAP50_95

# Evaluate model without ViM
print("Evaluating model without ViM")
metrics_without_vim = evaluate_model(model, test_loader)

# Evaluate model with ViM
print("Evaluating model with ViM")
metrics_with_vim = evaluate_model(model, test_loader, logit_bias, scale_factor=1.0)

# Save results to CSV
results = [metrics_without_vim, metrics_with_vim]
csv_file = 'comparison_results.csv'
csv_columns = ['precision', 'recall', 'mAP50', 'mAP50-95']

try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results:
            writer.writerow(data)
except IOError:
    print("I/O error")

# Save the trained model weights
torch.save(model.state_dict(), "trained_model_vim.pt")
