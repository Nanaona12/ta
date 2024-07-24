import os
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Path ke model yang telah disimpan
model_save_path = "D:/0000000000000000 OOD NOVIONA/Test/trained_model.pt"

# Muat model
model = YOLO(model="novi.yaml")
model.load(model_save_path)
model.eval()  # Set model ke mode evaluasi

# Transformasi gambar
transform = transforms.Compose([
    transforms.ToTensor()
])

# Fungsi untuk melakukan inferensi menggunakan model yang dimuat
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Tambahkan dimensi batch

    with torch.no_grad():
        results = model(image)
    
    return results

# Fungsi untuk menerapkan ViM (Virtual Logit Matching)
def apply_vim(results):
    vim_predictions = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:  # Pastikan ada prediksi
            # Ambil data dari bounding boxes
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            
            # Tambahkan bias pada confidence scores
            adjusted_confidences = confidences + 0.1
            
            # Gabungkan kembali dengan bounding boxes
            vim_predictions.append(np.hstack((boxes, adjusted_confidences.reshape(-1, 1))))
        else:
            vim_predictions.append(np.array([]))  # Jika tidak ada prediksi, kembalikan array kosong
    
    return vim_predictions

# Fungsi untuk membandingkan model dengan dan tanpa ViM
def compare_models(image_path):
    results = predict(image_path)
    
    # Periksa jika ada deteksi
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print("No detections found.")
        return

    vim_predictions = apply_vim(results)

    # Visualisasi hasil prediksi
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Tampilkan gambar asli
    image = Image.open(image_path).convert("RGB")
    ax[0].imshow(image)
    ax[0].set_title("Predictions without ViM")
    ax[1].imshow(image)
    ax[1].set_title("Predictions with ViM")

    # Tampilkan bounding boxes dan confidence scores
    for result in results:
        if result.boxes is not None:
            for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                ax[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
                ax[0].text(x1, y1, f'{conf:.2f}', color='white', fontsize=12, backgroundcolor='red')
    
    for vim_pred in vim_predictions:
        if vim_pred.size > 0:
            for box, conf in zip(vim_pred[:, :4], vim_pred[:, 4]):
                x1, y1, x2, y2 = box
                ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2))
                ax[1].text(x1, y1, f'{conf:.2f}', color='white', fontsize=12, backgroundcolor='green')

    plt.show()

# Contoh penggunaan
if __name__ == "__main__":
    image_path = 'D:/0000000000000000 OOD NOVIONA/Test/ghostnetwithv8v2.v1i.yolov8/valid/images/DJI_0004_JPG.rf.27fbef8f83697f6b7f32bb05b74e9b1b.jpg'  # Ganti dengan path ke gambar yang ingin Anda prediksi
    compare_models(image_path)
