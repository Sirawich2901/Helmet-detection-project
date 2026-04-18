from ultralytics import YOLO

# Import model
model = YOLO("C:/Users/tannytann/PycharmProjects/yolov11/yolo11s.pt")

# Predict
results = model("C:/Users/tannytann/PycharmProjects/yolov11/test_images", conf=0.5)  # ใช้ภาพหรือวิดีโอ

# Show Results
results.show()  # แสดงภาพหรือวิดีโอที่ตรวจจับแล้ว

# Save Results
results.save()  # บันทึกผลลัพธ์เป็นไฟล์
