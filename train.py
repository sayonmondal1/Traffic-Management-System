from ultralytics import YOLO
import torch
# model = YOLO("models/yolov8n.pt")
# model.train(data="D:/Programming/Python/Python Projects/Traffic_Management_System/data.yaml", epochs=200, device=0)
def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Using:", torch.cuda.get_device_name(0))
    model = YOLO("models/yolov8s.pt")
    # model.train(
    #     data="D:/Programming/Python/Python Projects/Traffic_Management_System/data.yaml",
    #     epochs=200,
    #     imgsz=640,
    #     batch=8,
    #     device=0,     
    #     workers=0  
    # )
    model.train(
        data="D:/Programming/Python/Python Projects/Traffic_Management_System/data.yaml",
        
        # Training params
        epochs=100,          # 200 is overkill for 10k images
        imgsz=832,
        batch=8,             # reduce if GPU crashes
        device=0,
        workers=0,

        # Accuracy boosters
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,

        # Regularization
        patience=20,
        augment=True,

        # Performance
        cache=False,
        amp=True,
        cos_lr=True
    )
if __name__ == "__main__":
    main()