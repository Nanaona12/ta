import os
import numpy as np
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Load the YOLOv8-Ghost model
model = YOLO(model="novi.yaml")

try:
    model.ckpt = torch.load("runs/detect/train22/weights/best.pt")
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Failed to load checkpoint: {e}")

# Check the contents of self.ckpt before saving
if model.ckpt is None:
    raise ValueError("Error: model.ckpt is None. Ensure the model has been trained properly.")
else:
    print("model.ckpt is populated.")

model_save_path = "D:/Test/runs/detect/train22/trained_model.pt"
model.save(model_save_path)

print(f"Model saved to {model_save_path}")