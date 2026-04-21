from ultralytics import YOLO
import torch
# model = YOLO("models/yolov8n.pt")
# model.train(data="D:/Programming/Python/Python Projects/Traffic_Management_System/data.yaml", epochs=200, device=0)
def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Using:", torch.cuda.get_device_name(0))
    model = YOLO("models/yolov8n.pt")
    model.train(
        data="D:/Programming/Python/Python Projects/Traffic_Management_System/data.yaml",
        epochs=200,
        imgsz=640,
        batch=8,
        device=0,     
        workers=0  
    )
if __name__ == "__main__":
    main()