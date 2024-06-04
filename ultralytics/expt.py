from ultralytics import YOLO
# load yolov8 segment model
model = YOLO("/data/wangshuai/warpyolo/yolov8/ultralytics/runs/detect/train/weights/best.pt")
# Use the model
success = model.export(format="onnx", opset=12, simplify=True)

