# คู่มือพารามิเตอร์ SAGE GeoClass V1.0.3

**ผู้พัฒนา:** Yutthana Thoenglom — Maejo University Phrae Campus

1. **Input raster:** เลือกชั้นข้อมูลปัจจัยที่ใช้จำแนก ตรวจสอบ CRS ขนาดพิกเซลและ NoData ให้ตรงกัน ข้อมูลประเภทชั้นดิน/การใช้ที่ดินควรใช้ nearest neighbor เมื่อปรับกริด
2. **Feature names / Registry:** ใส่ชื่อเรียงตามลำดับตัวแปรจริง JSON เป็นทางเลือก หากไม่ใช้ปล่อยว่าง
3. **Spectral roles:** กำหนดตำแหน่ง Blue, Green, Red, NIR แบบนับจาก 1 เท่านั้น เมื่อมีข้อมูลแบนด์ดังกล่าวจริง
4. **Training samples / Class field:** ใช้ Point/Polygon ที่มีค่าชั้นข้อมูลถูกต้อง และควบคุมความซ้ำซ้อนเชิงพื้นที่
5. **Validation strategy:** แนะนำ Group Split เพื่อลด leakage; validation fraction 0.2 และ independent test 0.2 เป็นค่าเริ่มต้น ต้องรักษาชุดทดสอบเป็นอิสระ
6. **Spatial block / Group K-fold:** ขนาดบล็อกใช้หน่วย CRS; 5 folds ต้องมีจำนวนกลุ่มและตัวอย่างแต่ละคลาสเพียงพอ
7. **Model selection:** Auto Compare เลือกโมเดลที่ใช้ได้ตาม dependencies ไม่รับประกันว่า 12 โมเดลถูกเปรียบเทียบด้วย Spatial CV เดียวกัน
8. **Spatial/Temporal CNN:** Spatial CNN ใช้ patch เช่น 15×15; Temporal CNN ต้องมีข้อมูลอย่างน้อย 2 ช่วงเวลาและเรียงเวลาอย่างถูกต้อง
9. **Trees, epochs, batch:** เพิ่มจำนวนจะเพิ่มเวลาหรือ RAM; เปิด Adaptive ML inference batch เมื่อเครื่องมี RAM จำกัด
10. **Feature importance / SHAP:** ใช้ตีความความสำคัญของตัวแปร ไม่ใช่หลักฐานเหตุและผล
11. **Post-filter / confidence:** Majority + Sieve อาจลบพื้นที่จริงขนาดเล็ก; threshold 0.6 ระบุพื้นที่ความเชื่อมั่นต่ำตามนิยามของอัลกอริทึม
12. **XAI / OOD / Reliability:** OOD ชี้ข้อมูลที่ต่างจากชุดฝึก; reliability เป็นตัวชี้วัดเชิง heuristic ไม่ใช่ calibrated probability
13. **Outputs:** เลือก Save ALL results to this directory เพื่อเก็บ CSV, raster, QA และ audit สำหรับงานวิจัย

## นโยบายเวอร์ชัน
V1 เป็นเวอร์ชันผลิตภัณฑ์; รหัส Frozen Scientific Core V2.5.5.4.3.5 และ Experimental Baseline V2.5.5.4.3.6 ยังคงเดิมทุกไบต์ ไม่มีการเปลี่ยนสูตร/โมเดล/parameter ID/default.
