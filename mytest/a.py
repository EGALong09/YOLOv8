from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 训练模型
train_results = model.train(
    data="coco8.yaml",  # 数据配置文件的路径
    epochs=100,  # 训练的轮数
    imgsz=640,  # 训练图像大小
    device="cpu",  # 运行的设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
)

# 在验证集上评估模型性能
metrics = model.val()

# 对图像进行目标检测
results = model("path/to/image.jpg")
results[0].show()

# 将模型导出为 ONNX 格式
path = model.export(format="onnx")  # 返回导出的模型路径from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 训练模型
train_results = model.train(
    data="coco8.yaml",  # 数据配置文件的路径
    epochs=100,  # 训练的轮数
    imgsz=640,  # 训练图像大小
    device="cpu",  # 运行的设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
)

# 在验证集上评估模型性能
metrics = model.val()

# 对图像进行目标检测
results = model("path/to/image.jpg")
results[0].show()

# 将模型导出为 ONNX 格式
path = model.export(format="onnx")  # 返回导出的模型路径