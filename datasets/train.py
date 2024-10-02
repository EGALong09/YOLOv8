from ultralytics import YOLO

model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    model.train(data='data.yaml',epochs=100,imgsz=640,workers=4,batch=4)
