from ultralytics import YOLO


# Load the pretrained model onto the selected device
model = YOLO('yolov8n-pose.pt')

# Train the model on the GPU
model.train(data='data.yaml', epochs=150, imgsz=640)