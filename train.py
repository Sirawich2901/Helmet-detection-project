from ultralytics import YOLO

# Import YOLO
model = YOLO("C:/Users/tannytann/PycharmProjects/yolov11/yolo11s.pt") #Path

# Trainmodel
model.train(
    data="data.yaml",          
    epochs=50,                 
    batch=32,                  # Batch size
    imgsz=640,                 # image size
    workers=4,                 # Worker
    project="C:/Users/tannytann/PycharmProjects/yolov11/runs/train",
    name="helmet_detection"    # Results folder name
)

#Show Results
results = model.val()  
print(results)
