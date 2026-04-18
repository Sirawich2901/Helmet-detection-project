from ultralytics import YOLO

# โหลดโมเดล YOLOv11
model = YOLO("C:/Users/tannytann/PycharmProjects/yolov11/yolo11s.pt") # สามารถเปลี่ยนเป็น yolov11s.pt, yolov11m.pt ตามความต้องการ

# เทรนโมเดล
model.train(
    data="data.yaml",     # ไฟล์ data.yaml
    epochs=50,                 # จำนวนรอบการเทรน
    batch=32,                  # ขนาด batch
    imgsz=640,                 # ขนาดของรูปภาพ (resolution)
    workers=4,                 # จำนวน workers สำหรับโหลดข้อมูล
    project="C:/Users/tannytann/PycharmProjects/yolov11/runs/train",
    name="helmet_detection"    # ชื่อการเทรน (บันทึกผลในโฟลเดอร์ runs/)
)

# แสดงผลการเทรน
results = model.val()  # ประเมินผลโมเดล
print(results)
