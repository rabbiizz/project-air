import time
import numpy as np
import cv2
from robot_control import MG400
import detector

def main():
    # 1. เชื่อมต่อหุ่นยนต์ MG400
    robot = MG400(ip="192.168.1.6")
    
    if not robot.use_real_robot:
        print("[ERROR] ไม่สามารถเชื่อมต่อกับหุ่นยนต์ได้")
        return
        
    print("[INFO] เชื่อมต่อและ Enable หุ่นยนต์สำเร็จ!")
    robot.set_speed(30)

    # กำหนดพิกัดจุด Home P5 [250, 0, 130, 0] และ Slot สำหรับการวางชิ้นงาน
    HOME_POSE = [250.0, 0.0, 130.0, 0.0]  
    SLOT_POSITIONS = {
        "Square":    [200.0, -150.0, 50.0, 0.0],  
        "Circle":    [200.0, -100.0, 50.0, 0.0],  
        "Trapezoid": [200.0, -50.0,  50.0, 0.0],  
        "Hexagon":   [200.0, 0.0,    50.0, 0.0]   
    }

    # กำหนดจุดกึ่งกลางภาพกล้อง (สำหรับวาดเป้าและคำนวณระยะทาง)
    IMG_CENTER_X, IMG_CENTER_Y = 320, 240

    # --- ใช้ Perspective Transform ชดเชยพิกัดจาก 4 จุด เพื่อความแม่นยำสูงสุด ---
    # จุดพิกัดจากกล้องที่อ่านได้ล่าสุด (Cam_X, Cam_Y)
    pts_cam = np.array([
        [119.7, 21.0],    # P1 (Square)
        [121.2, 112.5],   # P2 (Hexagon)
        [12.5, 28.9],     # P3 (Trapezoid)
        [-7.0, 122.5]     # P4 (Circle)
    ], dtype=np.float32)

    # จุดพิกัดจริงของหุ่นยนต์ (Robot_X, Robot_Y)
    pts_rob = np.array([
        [314.79, -66.27], # P1
        [225.58, -68.77], # P2
        [305.07, 33.942], # P3
        [215.59, 53.030]  # P4
    ], dtype=np.float32)

    # สร้างเมทริกซ์แปลงพิกัด (ดัดสเกล องศา และระยะทางให้ตรงกับหุ่นยนต์)
    align_matrix = cv2.getPerspectiveTransform(pts_cam, pts_rob)
    print("[INFO] สร้างเมทริกซ์แปลงพิกัด (4-Point Alignment) สำเร็จ")

    def get_shape_params(shape_name):
        if shape_name == "Square":      # P1
            return -137.2, -11.90
        elif shape_name == "Hexagon":   # P2
            return -134.1, -16.97
        elif shape_name == "Trapezoid": # P3
            return -136.5, 6.3346
        elif shape_name == "Circle":    # P4
            return -133.9, 13.800
        return 0.0, 0.0

    def apply_offset(shape_name, cam_x, cam_y):
        # แปลงพิกัดด้วยเมทริกซ์
        pt = np.array([[[cam_x, cam_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, align_matrix)
        real_x = float(transformed[0][0][0])
        real_y = float(transformed[0][0][1])
        target_z, target_r = get_shape_params(shape_name)
        return real_x, real_y, target_z, target_r

    try:
        robot.moveL(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3])
        time.sleep(1.5)
    except Exception as e:
        print(f"[ERROR] หุ่นยนต์ไม่สามารถเคลื่อนที่ได้ (ตรวจสอบ Limit): {e}")
        robot.close()
        return

    h_matrix = detector.load_calibration()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] ไม่สามารถเปิดกล้องได้")
        robot.close()
        return

    print("\n[INFO] เปิดหน้าต่างกล้อง... กำลังดึงข้อมูลจาก detector มาแสดงผล")
    print(">>> กดปุ่ม 's' บนคีย์บอร์ด (ขณะคลิกที่หน้าต่างกล้อง) เพื่อเริ่มหยิบวัตถุ")
    print(">>> กดปุ่ม 'q' เพื่อยกเลิกและปิดโปรแกรม\n")

    detected_objects = []

    while True:
        detected_objects = detector.scan_objects(cap, h_matrix)
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] อ่านภาพจากกล้องไม่ได้")
            break
            
        cv2.putText(frame, "Alignment: 4-Point Perspective Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # --- วาดเป้ากึ่งกลางกล้อง (Camera Center) ---
        cv2.line(frame, (IMG_CENTER_X - 15, IMG_CENTER_Y), (IMG_CENTER_X + 15, IMG_CENTER_Y), (0, 165, 255), 2)
        cv2.line(frame, (IMG_CENTER_X, IMG_CENTER_Y - 15), (IMG_CENTER_X, IMG_CENTER_Y + 15), (0, 165, 255), 2)
        cv2.circle(frame, (IMG_CENTER_X, IMG_CENTER_Y), 3, (0, 0, 255), -1)
        cv2.putText(frame, "Center", (IMG_CENTER_X + 10, IMG_CENTER_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        # -----------------------------------------
            
        for obj in detected_objects:
            cx, cy = obj['cx'], obj['cy']
            name = obj['name']
            x_mm, y_mm = obj['x'], obj['y']
            
            rx, ry, _, _ = apply_offset(name, x_mm, y_mm)
            
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{name}", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Robot X:{rx:.1f} Y:{ry:.1f}", (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        cv2.imshow("Camera Live", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print("[INFO] เริ่มกระบวนการหยิบวัตถุ...")
            break
        elif key == ord('q'):
            print("[INFO] ยกเลิกการทำงาน")
            cap.release()
            cv2.destroyAllWindows()
            robot.close()
            return

    if not detected_objects:
        print("[INFO] ไม่พบวัตถุใดๆ ในพื้นที่")
        cap.release()
        cv2.destroyAllWindows()
        robot.close()
        return

    for obj in detected_objects:
        rx, ry, rz, rr = apply_offset(obj['name'], obj['x'], obj['y'])
        obj['x'] = rx
        obj['y'] = ry
        obj['z_target'] = rz
        obj['angle'] = rr

    for obj in detected_objects:
        obj['distance'] = np.sqrt((obj['cx'] - IMG_CENTER_X)**2 + (obj['cy'] - IMG_CENTER_Y)**2)

    detected_objects = sorted(detected_objects, key=lambda k: k['distance'])
    print(f"[INFO] พบวัตถุทั้งหมด {len(detected_objects)} ชิ้น เรียงลำดับการหยิบเรียบร้อย")

    for idx, obj in enumerate(detected_objects):
        shape_name = obj['name']
        target_x = obj['x']
        target_y = obj['y']
        Z_PICK = obj['z_target']
        target_angle = obj['angle']
        Z_HOVER_3CM = Z_PICK + 30.0  
        
        print(f"\n--- กำลังหยิบชิ้นที่ {idx+1}: {shape_name} ที่พิกัด X={target_x:.2f}, Y={target_y:.2f} ---")

        while True:
            current_objs = detector.scan_objects(cap, h_matrix)
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "Alignment: 4-Point Perspective Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # --- วาดเป้ากึ่งกลางกล้อง (Camera Center) ---
                cv2.line(frame, (IMG_CENTER_X - 15, IMG_CENTER_Y), (IMG_CENTER_X + 15, IMG_CENTER_Y), (0, 165, 255), 2)
                cv2.line(frame, (IMG_CENTER_X, IMG_CENTER_Y - 15), (IMG_CENTER_X, IMG_CENTER_Y + 15), (0, 165, 255), 2)
                cv2.circle(frame, (IMG_CENTER_X, IMG_CENTER_Y), 3, (0, 0, 255), -1)
                
                for c_obj in current_objs:
                    cv2.circle(frame, (c_obj['cx'], c_obj['cy']), 5, (255, 0, 0), -1)
                cv2.imshow("Camera Live", frame)
                cv2.waitKey(1)
                
            matched_obj = None
            min_d = float('inf')
            
            for c_obj in current_objs:
                cx_t, cy_t, _, _ = apply_offset(c_obj['name'], c_obj['x'], c_obj['y'])
                d = np.sqrt((cx_t - target_x)**2 + (cy_t - target_y)**2)
                if d < min_d:
                    min_d = d
                    matched_obj = c_obj

            if matched_obj:
                err_x = matched_obj['cx'] - IMG_CENTER_X
                err_y = matched_obj['cy'] - IMG_CENTER_Y
                
                if abs(err_x) <= 1 and abs(err_y) <= 1:
                    print(f"[INFO] แมตช์จุดศูนย์กลางสำเร็จ")
                    target_x, target_y, Z_PICK, target_angle = apply_offset(matched_obj['name'], matched_obj['x'], matched_obj['y'])
                    Z_HOVER_3CM = Z_PICK + 30.0
                    break
            
            break  
            
        robot.moveL(target_x, target_y, HOME_POSE[2], target_angle)
        time.sleep(0.5)

        robot.moveL(target_x, target_y, Z_HOVER_3CM, target_angle)
        time.sleep(0.3)

        robot.moveL(target_x, target_y, Z_PICK, target_angle)
        time.sleep(0.3)

        robot.set_gripper(1)
        time.sleep(1.0) 

        robot.moveL(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3])
        time.sleep(0.1)

        if shape_name in SLOT_POSITIONS:
            slot_x, slot_y, slot_z, slot_r = SLOT_POSITIONS[shape_name]

            robot.moveL(slot_x, slot_y, slot_z + 30.0, slot_r)
            time.sleep(1.0) 

            robot.moveL(slot_x, slot_y, slot_z, slot_r)
            time.sleep(0.5)

            robot.set_gripper(0)
            time.sleep(1.0)

            robot.moveL(slot_x, slot_y, slot_z + 30.0, slot_r)
            time.sleep(0.3)

        robot.moveL(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3])
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    robot.close()
    print("[INFO] ทำงานครบทุกชิ้นและปิดระบบเรียบร้อยครับ")

if __name__ == "__main__":
    main()