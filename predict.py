from ultralytics import YOLO

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = YOLO("C:/Users/tannytann/PycharmProjects/yolov11/yolo11s.pt")

# ทำการทดสอบ (ปรับพาธไฟล์ตามที่ต้องการ)
results = model("C:/Users/tannytann/PycharmProjects/yolov11/test_images", conf=0.5)  # ใช้ภาพหรือวิดีโอ

# แสดงผลลัพธ์
results.show()  # แสดงภาพหรือวิดีโอที่ตรวจจับแล้ว

# บันทึกผลลัพธ์
results.save()  # บันทึกผลลัพธ์เป็นไฟล์
