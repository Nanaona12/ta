import torch
from ultralytics import YOLO

# Path ke model yang telah disimpan
model_save_path = "D:/Test/runs/detect/train22/trained_model.pt"

# Muat model
model = YOLO(model="novi.yaml")
model.load(model_save_path)
# model.eval()  # Set model ke mode evaluasi

# # Fungsi untuk melakukan inferensi menggunakan model yang dimuat
# def predict(image_path):
#     from PIL import Image
#     from torchvision import transforms

#     # Transformasi gambar
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     # Buka gambar
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0)  # Tambahkan dimensi batch

#     # Lakukan inferensi
#     with torch.no_grad():
#         predictions = model(image)
    
#     return predictions

# Contoh penggunaan
# image_path = 'path/to/your/image.jpg'  # Ganti dengan path ke gambar yang ingin Anda prediksi
# predictions = predict(image_path)
# print(predictions)
