from ultralytics import YOLO

# Import YOLO
model = YOLO("C:/Users/tannytann/PycharmProjects/yolov11/yolo11s.pt") #Path

# Trainmodel
model.train(
    data="data.yaml",     # ไฟล์ data.yaml
    epochs=50,                 # จำนวนรอบการเทรน
    batch=32,                  # ขนาด batch
    imgsz=640,                 # ขนาดของรูปภาพ (resolution)
    workers=4,                 # จำนวน workers สำหรับโหลดข้อมูล
    project="C:/Users/tannytann/PycharmProjects/yolov11/runs/train",
    name="helmet_detection"    # ชื่อการเทรน (บันทึกผลในโฟลเดอร์ runs/)
)

#Show Results
results = model.val()  # ประเมินผลโมเดล
print(results)
