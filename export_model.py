from ultralytics import YOLO

YOLO_MODEL = "yolov8n.pt"  


#to export to ONNX model
# model= YOLO(YOLO_MODEL)
# model.export(
#         format="onnx",          
#         dynamic=True,           
#         simplify=True,         
#         nms=True                
# )

#to export to OpenVINO
model= YOLO(YOLO_MODEL)
model.export(
    format="openvino",
    imgsz=(384, 640)
)