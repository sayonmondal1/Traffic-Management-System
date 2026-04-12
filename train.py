from ultralytics import YOLO
# Load model and dataset
model = YOLO("models/yolov8n.pt")
model.train(data="D:/Programming/Python/Python Projects/Traffic_Management_System/data.yaml", epochs=200, device="cpu")
"""
import torch
print("Using device:", torch.cuda.get_device_name(0))
model = YOLO("yolov8n.pt")
model.train(data=r"D:/Programming/Python/Python Projects/Traffic_Management_System/data.yaml", epochs=200, device=0)  # instead of "cuda"
"""