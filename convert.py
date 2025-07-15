from ultralytics import YOLO
model = YOLO(r"C:\Users\OS\Desktop\gs25\weights\last_yolov11.pt")
model.export(format="onnx", dynamic=False, simplify=False)