SAGE GeoClass QGIS Plugin V1.0.0

ติดตั้งผ่าน QGIS > Plugins > Manage and Install Plugins > Install from ZIP
เลือกไฟล์ SAGE_GeoClass_QGIS_Plugin_V1_0_0.zip จากนั้นเปิดใช้งานปลั๊กอิน
เรียกใช้ Processing Toolbox > SAGE GeoClass > SAGE-GeoClass V2.5.5.4.3.6

ต้องมี Python dependencies ของโมเดลที่เลือกอยู่ใน QGIS Python environment อยู่แล้ว
ปลั๊กอินไม่ติดตั้ง dependency โดยอัตโนมัติ และไม่ได้รับรองผลรันจริงใน QGIS 4.2.2
หากเปิดไม่สำเร็จ ให้ดู QGIS Python error log และตรวจสอบ dependency
Frozen Core และ Experimental Baseline เก็บไว้โดยไม่แก้ไข

Plugin V1.0.1: เปลี่ยนชื่อแสดงผลใน Processing Toolbox เป็น SAGE-GeoClass V1 — Scientific Guard (Frozen Core) เท่านั้น
Scientific modules และ algorithm ID เดิมไม่เปลี่ยนแปลง เพื่อรักษาความเข้ากันได้
